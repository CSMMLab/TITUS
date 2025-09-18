include("Problem.jl")
include("TTN.jl")

using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
np = pyimport("numpy")

struct BUGIntegrator
    # spatial grid of cell interfaces
    x::Array{Float64};

    Δt::Float64;

    # Solver settings
    settings::Settings;

    problem::Problem;
    rhs::Rhs;

    # constructor
    function BUGIntegrator(settings)
        x = settings.x;
        Δt = settings.Δt;

        new(x,Δt,settings,Problem(settings),Rhs(settings));
    end
end

function Solve(obj::BUGIntegrator)

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
        _, Ĉ¹, Û¹ = Step(obj, Y, FY)
        Y1 = TTN(Y.id, Û¹, Ĉ¹,ones(1,1),ones(1,1))
        Y = θ(obj, Y1)
        next!(prog) # update progress bar
    end
    return Y, rVec
end

# update and augment the ith basis matrix
function Φ(obj::BUGIntegrator, Y::TTN, FY::Vector{TTN}, i::Int)
    id = Y.leaves[i].id
    r = size(Y.C);
    hatr = [2*rᵢ for rᵢ in r]
    hatr[1] = Int(0.5 * hatr[1])
    
    # perform prolongation and retraction
    _, FᵢY = prolong_and_retract_full(FY, Y, i)

    if length(Y.leaves[i].leaves) == 0 # Yᵢ is leaf
        U⁰ = Y.leaves[i].C
        K⁰ = U⁰*Y.leaves[i].S;
        K¹ = K⁰

        # define right-hand side of Φ step
        FΦ = K -> begin
            rhsK = zeros(size(K))
            for (k, FᵢYₖ) in enumerate(FᵢY) 
                rhsK .+= obj.rhs.A[k][Y.leaves[i].id]*K*FᵢYₖ.VᵀFV';
            end
            return rhsK
        end
        K¹ = rk(FΦ, K⁰, obj.Δt)
        Û¹,_ = np.linalg.qr([K¹ U⁰], mode="reduced");
        X̂ᵢ = TTN(id, [], Û¹)
        M = Û¹'*U⁰
    else # Yᵢ is not a leaf
        Ĉ⁰, Ĉ¹, Û¹ = Step(obj, Y.leaves[i], FᵢY) # Yᵢ should be the same as Y.leaves[i]
        Q,_ = np.linalg.qr([tenmat(Ĉ¹, 1)' tenmat(Ĉ⁰, 1)'], mode="reduced"); 
        hatr = [rᵢ for rᵢ in size(Ĉ⁰)]; hatr[1] = 2*hatr[1];
        X̂ᵢ = TTN(id, Û¹, matten(Q', 1, hatr))
        M = inner(X̂ᵢ, Y.leaves[i]); # Y.leaves[i] has old basis
    end
    
    return X̂ᵢ, M
end

# augment and update core tensor
function Ψ(obj::BUGIntegrator, Y::TTN, M::Vector{Matrix{Float64}}, FY::Vector{TTN}, Δt::Float64, X̂::Vector{TTN})
    Ĉ⁰ = ttm(Y.C, M, 2:length(Y.leaves)+1)
    hatY = TTN(Y.id, X̂, Ĉ⁰, Y.S, Y.VᵀFV)
    FhatY = eval(obj.rhs, hatY)
    Ĉ¹ = deepcopy(Ĉ⁰)

    UᵀAU = Vector{Matrix}[]
    for (i, FhatYᵢ) in enumerate(FhatY)
        push!(UᵀAU, Matrix[])
        for (leaf, X̂ᵢ) in zip(FhatYᵢ.leaves, X̂)
            push!(UᵀAU[i], inner(X̂ᵢ, leaf))
        end
    end

    FΨ = C -> begin
        dC = zeros(size(Ĉ⁰))
        for (i, FYᵢ) in enumerate(FY)
            dCᵢ = ttm(C, FYᵢ.VᵀFV, 1)
            for j in eachindex(UᵀAU[i])
                dCᵢ = ttm(dCᵢ, UᵀAU[i][j], j+1)
            end
            dC += dCᵢ
        end
        return dC
    end

    C0 = ttm(Ĉ⁰, FY[1].S, 1) # S is always the same
    Ĉ¹ = rk(FΨ, C0, Δt)

    return Ĉ⁰, Ĉ¹
end

# Algorithm 4: Rank-augmenting TTN integrator
function Step(obj::BUGIntegrator, Y::TTN, FY::Vector{TTN})
    Û¹ = TTN[];
    M = Matrix{Float64}[];

    for i in 1:length(Y.leaves)
        Ûᵢ, Mᵢ = Φ(obj, Y, FY, i)
        push!(Û¹, Ûᵢ); push!(M, Mᵢ);
    end
    Ĉ⁰, Ĉ¹ = Ψ(obj, Y, M, FY, obj.settings.Δt, Û¹);

    return Ĉ⁰, Ĉ¹, Û¹
end

function θ(obj::BUGIntegrator, Y::TTN)
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