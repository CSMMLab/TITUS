using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
np = pyimport("numpy")

struct ParallelIntegratorIneff
    # spatial grid of cell interfaces
    x::Array{Float64};

    Δt::Float64;

    # Solver settings
    settings::Settings;

    problem::Problem;
    rhs::Rhs;

    # constructor
    function ParallelIntegratorIneff(settings::Settings)
        x = settings.x;
        Δt = settings.Δt;
        new(x,Δt,settings,Problem(settings),Rhs(settings));
    end
end

function Solve(obj::ParallelIntegratorIneff)

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
    prog = Progress(nt,1)

    for m = 1:nt
        FY = eval(rhs, Y)
        _, Ĉ¹, Û¹ = Step(obj, Y, FY)
        Y1 = TTN(Y.id, Û¹, Ĉ¹,ones(1,1),ones(1,1))
        Y = θ(obj, Y1)
        next!(prog) # update progress bar
    end
    return Y
end

# update and augment the ith basis matrix
function Φ(obj::ParallelIntegratorIneff, Y::TTN, FY::Vector{TTN}, i::Int)
    id = Y.leaves[i].id
    r = size(Y.C);
    hatr = [2*rᵢ for rᵢ in r]
    hatr[1] = Int(0.5 * hatr[1])

    hatXᵢ = copy_subtree(Y.leaves[i])

    # perform prolongation and retraction
    Sᵀ, FᵢY = prolong_and_retract_full(FY, Y, i)

    if length(Y.leaves[i].leaves) == 0 # Yᵢ is leaf
        U⁰ = Y.leaves[i].C
        K⁰ = U⁰*Sᵀ';
        # define right-hand side of Ψ step
        FΦ = K -> begin
            rhsK = zeros(size(K))
            for (k, FᵢYₖ) in enumerate(FᵢY) 
                rhsK .+= obj.rhs.A[k][Y.leaves[i].id]*K*FᵢYₖ.VᵀFV';
            end
            return rhsK
        end
        K¹ = rk(FΦ, K⁰, obj.Δt)

        Û¹,_ = np.linalg.qr([U⁰ K¹], mode="reduced");
        Û¹[:, 1:r[i+1]] .= U⁰
        hatXᵢ.C = Û¹
        X̃ᵢ = TTN(id, [], Û¹[:, (r[i+1]+1):end]);
    else # Yᵢ is not a leaf
        Ĉ⁰, Ĉ¹, Û¹ = Step(obj, Y.leaves[i], FᵢY) # Yᵢ should be the same as Y.leaves[i]
        Q,_ = np.linalg.qr([tenmat(Ĉ⁰, 1)' tenmat(Ĉ¹, 1)'], mode="reduced"); 
        Q[:, 1:size(Ĉ⁰, 1)] .= tenmat(Ĉ⁰, 1)'
        hatr = [rᵢ for rᵢ in size(Ĉ⁰)]; 
        X̃ᵢ = TTN(id, Û¹, matten(Q[:, (size(Ĉ⁰, 1)+1):end]', 1, hatr))
        hatr[1] = 2*hatr[1];
        hatXᵢ = TTN(id, Û¹, matten(Q', 1, hatr))
    end
    
    return hatXᵢ, X̃ᵢ
end

# update core tensor
function Ψ1st(obj::ParallelIntegratorIneff, C¹::Array, FY::Vector{TTN}, Δt::Float64, X::Vector{TTN})
    for FYᵢ in FY
        dC = deepcopy(ttm(FYᵢ.C, FYᵢ.VᵀFV*FYᵢ.S, 1))
        for (i, (leaf, Xᵢ)) in enumerate(zip(FYᵢ.leaves, X))
            UᵀAU = inner(Xᵢ, leaf)
            dC = ttm(dC, UᵀAU, i+1)
        end
        C¹ .= C¹ .+ Δt * dC
    end

    return C¹
end

# update core tensor
function Ψ(obj::ParallelIntegratorIneff, C¹::Array, FY::Vector{TTN}, Δt::Float64, X::Vector{TTN})
    UᵀAU = Vector{Matrix}[]
    for (i, FYᵢ) in enumerate(FY)
        push!(UᵀAU, Matrix[])
        for (leaf, Xᵢ) in zip(FYᵢ.leaves, X)
            push!(UᵀAU[i], inner(Xᵢ, leaf))
        end
    end

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
    
    C0 = ttm(C¹, FY[1].S, 1) # S is always the same
    return rk(FΨ, C0, Δt)
end

# Algorithm 4: Rank-augmenting TTN integrator
function Step(obj::ParallelIntegratorIneff, Y::TTN, FY::Vector{TTN})
    Û¹ = TTN[];

    FYtmp = copy(FY) # not needed

    r = [size(Y.C, i) for i in 1:length(size(Y.C))]

    rField = []
    for i = 1:length(size(Y.C))
        push!(rField,1:r[i])
    end
    
    rHat = deepcopy(r)
    rHat[2:end] .= 2 * rHat[2:end]

    # create tensor with double dimension
    Ĉ¹ = matten(zeros(Float64, r[1], prod(2 .* r[2:end])),1,rHat)
    Ĉ⁰ = matten(zeros(Float64, r[1], prod(2 .* r[2:end])),1,rHat)
    Ĉ⁰[CartesianIndices(Tuple(rField))] .= Y.C;

    Ĉ¹[CartesianIndices(Tuple(rField))] = Ψ(obj, Y.C, FYtmp, obj.settings.Δt, Y.leaves);
    for i in 1:length(Y.leaves)
        hatXᵢ, X̃ᵢ = Φ(obj, Y, FY, i);
        push!(Û¹, hatXᵢ);

        X̃ = deepcopy(Y.leaves); X̃[i] = X̃ᵢ;
        rFieldᵢ = deepcopy(rField); rFieldᵢ[i + 1] = rField[i + 1] .+ r[i + 1]
        Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = Ψ1st(obj, zeros(size(Y.C)), FYtmp, obj.settings.Δt, X̃);
    end

    return Ĉ⁰, Ĉ¹, Û¹ 
end

function θ(obj::ParallelIntegratorIneff, Y::TTN)
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