using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
using Base.Threads
np = pyimport("numpy")

mutable struct ParallelIntegrator
    # spatial grid of cell interfaces
    x::Array{Float64};

    Δt::Float64;

    # Solver settings
    settings::Settings;

    problem::Problem;
    rhs::Rhs;

    barC::Array{Array}
    Û¹::Array{Matrix{Float64}}

    nodes::Vector{TTN}
    leaves::Vector{TTN}

    FYNodes::Array{Vector{TTN}}
    FYLeaves::Array{Vector{TTN}}

    Y0::TTN

    # constructor
    function ParallelIntegrator(settings::Settings)
        x = settings.x;
        Δt = settings.Δt;
        rhs = Rhs(settings)
        nLeaves = length(rhs.A[1])
        nNodes = nLeaves - 1
        barC = Array{Array}(undef, nNodes)
        Û¹ = Array{Matrix{Float64}}(undef, nLeaves)
        FYNodes = Array{Vector{TTN}}(undef, nNodes)
        FYLeaves = Array{Vector{TTN}}(undef, nLeaves)
        new(x,Δt,settings,Problem(settings),rhs,barC,Û¹,[],[],FYNodes,FYLeaves);
    end
end

function Solve(obj::ParallelIntegrator)

    r = obj.settings.r
    n = obj.settings.NCells
    rhs = obj.rhs
    if obj.settings.problem == "radiation"
        Y = generateSmallRadTree(obj.settings, rhs)
    elseif obj.settings.problem == "radiationUQ"
        Y = generateRadTree(obj.settings, rhs)
    elseif obj.settings.problem == "radiationUQ8D"
        Y = generateRadTree8D(obj.settings)
    elseif obj.settings.problem == "IsingModel"
        Y = generateIsingTree(obj.settings)
    elseif obj.settings.problem == "radiation2DUQ" || obj.settings.problem == "Lattice"
        Y = generateRadTree2D(obj.settings)
    else
        Y = generateOrthTestTree(r, n)
    end
    
    Δt = obj.settings.Δt
    nt = Int(floor(obj.settings.tEnd / Δt))
    rVec = zeros(nt, length(get_rank(Y)))
    prog = Progress(nt,1)

    for m = 1:nt
        rVec[m, :] .= get_rank(Y)
        FY = eval(rhs, Y)
        Y1 = Step(obj, Y, FY)
        Y = θ(obj, Y1)
        next!(prog) # update progress bar
    end
    return Y, rVec
end

# Algorithm 4: Rank-augmenting TTN integrator
function buildF(obj::ParallelIntegrator, Y::TTN, FY::Vector{TTN})
    for i in 1:length(Y.leaves)
        build(obj, Y, FY, i);
    end
end

# update and augment the ith basis matrix
function build(obj::ParallelIntegrator, Y::TTN, FY::Vector{TTN}, i::Int)

    _, FᵢY = prolong_and_retract_full(FY, Y, i)

    if length(Y.leaves[i].leaves) == 0 # Yᵢ is leaf
        obj.FYLeaves[Y.leaves[i].id] = FᵢY
    else # Yᵢ is not a leaf
        obj.FYNodes[Y.leaves[i].id] = FᵢY
        buildF(obj, Y.leaves[i], FᵢY) # Yᵢ should be the same as Y.leaves[i]
    end
end

function aug(obj::ParallelIntegrator, Y::TTN, Y0::TTN, i::Int)
    id = Y.leaves[i].id
    hatXᵢ = Y.leaves[i]#copy_subtree(Y.leaves[i])
    if length(Y.leaves[i].leaves) == 0 # Y is leaf
        r = size(Y.leaves[i].C, 2)
        Y.leaves[i].C = obj.Û¹[id];
        hatXᵢ.C = obj.Û¹[id]
        X̃ᵢ = TTN(id, [], obj.Û¹[id][:, (r+1):end]);
    else # Yᵢ is not a leaf
        Ĉ⁰, Ĉᵢ, Û¹ = Augment(obj, Y.leaves[i], Y0.leaves[i]);
        Q,_ = np.linalg.qr([tenmat(Ĉ⁰, 1)' tenmat(Ĉᵢ, 1)'], mode="reduced"); 
        Q[:, 1:size(Ĉ⁰, 1)] .= tenmat(Ĉ⁰, 1)';
        hatr = [rᵢ for rᵢ in size(Ĉ⁰)]; 
        X̃ᵢ = TTN(id, Û¹, matten(Q[:, (size(Ĉ⁰, 1)+1):end]', 1, hatr));
        hatr[1] = 2*hatr[1];
        hatXᵢ = TTN(id, Û¹, matten(Q', 1, hatr));
        Y.leaves[i].C = matten(Q', 1, hatr);
    end
    return X̃ᵢ, hatXᵢ
end

function Augment(obj::ParallelIntegrator, Y::TTN, Y0::TTN)
    Û¹ = TTN[];

    r = [size(Y.C, i) for i in 1:length(size(Y.C))];

    rField = []
    for i = 1:length(size(Y.C))
        push!(rField,1:r[i]);
    end
    
    rHat = deepcopy(r)
    rHat[2:end] .= 2 * rHat[2:end];

    # determine Ĉ⁰ for correct augmentation
    Ĉ¹ = matten(zeros(Float64, r[1], prod(2 .* r[2:end])),1,rHat)
    Ĉ⁰ = matten(zeros(Float64, r[1], prod(2 .* r[2:end])),1,rHat);

    Ĉ⁰[CartesianIndices(Tuple(rField))] .= Y.C;
    for i in eachindex(Y.leaves)
        X̃ᵢ, hatXᵢ = aug(obj, Y, Y0, i);
        push!(Û¹, hatXᵢ);
        X̃ = deepcopy(Y0.leaves); X̃[i] = X̃ᵢ;
        rFieldᵢ = deepcopy(rField); rFieldᵢ[i + 1] = rField[i + 1] .+ r[i + 1];
        Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = Ψ1st(obj, zeros(size(Y.C)), obj.FYNodes[Y.id], X̃);
    end
    Ĉ¹[CartesianIndices(Tuple(rField))] = obj.barC[Y.id]
    
    return Ĉ⁰, Ĉ¹, Û¹
end

function StepParallel(obj::ParallelIntegrator, Y::TTN, FY::Vector{TTN})

    obj.nodes = get_connecting_nodes(Y)
    obj.leaves = get_leaf_nodes(Y)

    obj.FYNodes[Y.id] = FY
    buildF(obj, Y, FY)

    @sync begin
        @async begin
            @threads for node in obj.nodes
                obj.barC[node.id] = Ψ(obj, deepcopy(node.C), obj.FYNodes[node.id], node.leaves)
            end
        end

        @async begin
            @threads for leaf in obj.leaves
                obj.Û¹[leaf.id] = Φ(obj, leaf, obj.FYLeaves[leaf.id])
            end
        end
    end

    # glue together the tree
    _, Ĉ¹, Û¹ = Augment(obj, Y, copy_subtree(Y))

    return TTN(Y.id, Û¹, Ĉ¹, ones(1,1), ones(1,1))
end

function Step(obj::ParallelIntegrator, Y::TTN, FY::Vector{TTN})
    
    obj.nodes = get_connecting_nodes(Y)
    obj.leaves = get_leaf_nodes(Y)

    obj.FYNodes[Y.id] = FY
    buildF(obj, Y, FY)

    for node in obj.nodes
        obj.barC[node.id] = Ψ(obj, deepcopy(node.C), obj.FYNodes[node.id], node.leaves)
    end

    for leaf in obj.leaves
        obj.Û¹[leaf.id] = Φ(obj, leaf, obj.FYLeaves[leaf.id])
    end

    # glue together the tree
    _, Ĉ¹, Û¹ = Augment(obj, Y, copy_subtree(Y))

    return TTN(Y.id, Û¹, Ĉ¹, ones(1,1), ones(1,1))
end

function Φ(obj::ParallelIntegrator, Y::TTN, FY::Vector{TTN})
    U⁰ = Y.C
    r = size(U⁰, 2)

    # define right-hand side of the Ψ step
    FΦ = K -> begin
        rhsK = zeros(size(K))
        for (k, FYₖ) in enumerate(FY) 
            rhsK .+= obj.rhs.A[k][Y.id]*K*FYₖ.VᵀFV';
        end
        return rhsK
    end

    K⁰ = U⁰*Y.S;
    K¹ = rk(FΦ, K⁰, obj.Δt)

    Û¹,_ = qr([U⁰ K¹]); Û¹ = Matrix(Û¹[:, 1:2*size(U⁰,2)])#np.linalg.qr([U⁰ K¹], mode="reduced");
    Û¹[:, 1:r] .= U⁰
    return Û¹
end

# update core tensor
function Ψ1st(obj::ParallelIntegrator, C¹::Array, FY::Vector{TTN}, X::Vector{TTN})
    for FYᵢ in FY
        dC = ttm(FYᵢ.C, FYᵢ.VᵀFV*FYᵢ.S, 1)
        for (i, (leaf, Xᵢ)) in enumerate(zip(FYᵢ.leaves, X))
            UᵀAU = inner(Xᵢ, leaf)
            dC = ttm(dC, UᵀAU, i+1)
        end
        C¹ .= C¹ .+ obj.Δt * dC
    end

    return C¹
end


"""
Update the core tensor 
# Arguments
- `obj::ParallelIntegrator`: Reference to the Solver.
- `C¹::Array`: Starting value of core tensor. This is the root to a tree τ.
- `FY::Vector{TTN}`: TTNO at current solution for the tree τ.
- `X::Vector{TTN}`: All basis functions of the root. If τ has n leaves, then length(X) = n.
"""
function Ψ(obj::ParallelIntegrator, C¹::Array, FY::Vector{TTN}, X::Vector{TTN})

    # For every leaf in FY, compute the projection onto the basis X.
    UᵀAU = Vector{Matrix}[]
    for (i, FYᵢ) in enumerate(FY)
        push!(UᵀAU, Matrix[])
        for (leaf, Xᵢ) in zip(FYᵢ.leaves, X)
            push!(UᵀAU[i], inner(Xᵢ, leaf))
        end
    end

    # Define right-hand side of ODE to evolve C. The part above the current node has already been projected to an r₀ x r₀
    # matrix which is stored in FYᵢ.VᵀFV. The leaves of the current node have been projected onto the basis X before and are
    # stored in UᵀAU
    FΨ = C -> begin
        dC = zeros(size(C))
        for (i, FYᵢ) in enumerate(FY)
            dCᵢ = ttm(C, FYᵢ.VᵀFV, 1)
            for j in eachindex(UᵀAU[i])
                dCᵢ = ttm(dCᵢ, UᵀAU[i][j], j+1)
            end
            dC += dCᵢ
        end
        return dC
    end
    
    # The initial condition needs to be multiplied with the information gathered while passing information from the root of
    # the entire tree to the current node. This is stored in FY[i].S for all i and is always the same.
    C0 = ttm(C¹, FY[1].S, 1) # S is always the same
    return rk(FΨ, C0, obj.Δt)
end

function θ(obj::ParallelIntegrator, Y::TTN)
    P = Array{Float64, 2}[]
    for i in 1:length(Y.leaves)
        Pᵢ, Σ, _ = svd(tenmat(Y.C, i+1))
        rMaxTotal = obj.settings.rMax;
        rMinTotal = obj.settings.rMin;

        tmp = 0.0;
        tol = obj.settings.ϵ * norm(Σ);

        rmax = Int(floor(size(Σ,1)/2));

        for j=1:2*rmax
            tmp = sqrt(sum(Σ[j:2*rmax]).^2);
            if tmp < tol
                rmax = j;
                break;
            end
        end

        # if 2*r was actually not enough move to highest possible rank
        if rmax == -1
            rmax = rMaxTotal;
        end

        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,rMinTotal);

        Pᵢ = Pᵢ[:,1:rmax]
        push!(P, Pᵢ')

        if length(Y.leaves[i].leaves) == 0 # Yᵢ is leaf
            Y.leaves[i].C = Y.leaves[i].C*Pᵢ
        else
            Yᵢ = copy_subtree(Y.leaves[i])
            Yᵢ.C = ttm(Yᵢ.C, Pᵢ', 1)
            Y.leaves[i] = θ(obj, Yᵢ)
        end
    end
    Y.C = ttm(Y.C, P, [i+1 for i in 1:length(Y.leaves)])
    return Y
end