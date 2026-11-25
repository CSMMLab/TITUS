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

    rhs::Rhs;

    # constructor
    function BUGIntegrator(settings)
        x = settings.x;
        Δt = settings.Δt;

        new(x,Δt,settings,Rhs(settings));
    end
end

function Solve(obj::BUGIntegrator)

    r = obj.settings.r
    n = obj.settings.NCells
    rhs = obj.rhs
    if obj.settings.problem == "radiation"
        Y = generateSmallRadTree(obj.settings)
    elseif obj.settings.problem == "radiationUQ"
        Y = generateRadTree(obj.settings)
    elseif obj.settings.problem == "radiationUQ8D"
        Y = generateRadTree8D(obj.settings)
    elseif obj.settings.problem == "IsingModel"
        Y = generateIsingTree(obj.settings)
    elseif obj.settings.problem == "radiation2DUQ" || obj.settings.problem == "Lattice" || obj.settings.problem == "ProtonUQ"
        Y = generateRadTree2D(obj.settings)
    elseif obj.settings.problem == "radiation3DUQ"
        Y = generateRadTree3D(obj.settings)
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
    nF = length(obj.rhs.A) # entries nF+1:end of FY are source terms
    
    # perform prolongation and retraction
    _, FᵢY = prolong_and_retract_full(FY, Y, i, nF)

    if length(Y.leaves[i].leaves) == 0 # Yᵢ is leaf
        U⁰ = Y.leaves[i].C
        K⁰ = U⁰*Y.leaves[i].S;

        # source contribution to the K-step: U_Q,ᵢ (VᵀQV)ᵀ, independent of K
        srcK = zeros(size(K⁰))
        for k in nF+1:length(FᵢY)
            srcK .+= FᵢY[k].C * FᵢY[k].VᵀFV'
        end

        # define right-hand side of Φ step
        FΦ = K -> begin
            rhsK = Base.copy(srcK)
            for k in 1:nF
                rhsK .+= obj.rhs.A[k][id]*K*FᵢY[k].VᵀFV';
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
        r̂ = [rᵢ for rᵢ in size(Ĉ⁰)]; r̂[1] = size(Q, 2); # = 2rᵢ unless limited by the children ranks
        X̂ᵢ = TTN(id, Û¹, matten(Q', 1, r̂))
        M = inner(X̂ᵢ, Y.leaves[i]); # Y.leaves[i] has old basis
    end
    
    return X̂ᵢ, M
end

# augment and update core tensor
function Ψ(obj::BUGIntegrator, Y::TTN, M::Vector{Matrix{Float64}}, FY::Vector{TTN}, Δt::Float64, X̂::Vector{TTN})
    nF = length(obj.rhs.A) # entries nF+1:end of FY are source terms
    Ĉ⁰ = ttm(Y.C, M, 2:length(Y.leaves)+1)
    hatY = TTN(Y.id, X̂, Ĉ⁰, Y.S, Y.VᵀFV)
    FhatY = eval(obj.rhs, hatY)[1:nF] # flux terms evaluated in the augmented basis

    # Galerkin matrices X̂ᵀAX̂ of the flux terms
    UᵀAU = Vector{Matrix}[]
    for FhatYᵢ in FhatY
        push!(UᵀAU, Matrix[inner(X̂ᵢ, FhatX̂ᵢ) for (FhatX̂ᵢ, X̂ᵢ) in zip(FhatYᵢ.leaves, X̂)])
    end

    # Galerkin projection of the source terms, C_Q ×₀ (VᵀFV S) ×ⱼ ⟨X̂ⱼ, U_Q,ⱼ⟩ (constant in C)
    srcC = zeros(size(Ĉ⁰))
    for k in nF+1:length(FY)
        FQ = copy_subtree(FY[k])
        FQ.C = ttm(FQ.C, FQ.VᵀFV * FQ.S, 1) # multiply the projected part to mode 0
        srcC .+= project_and_eval(FQ, X̂)
    end

    FΨ = C -> begin
        dC = Base.copy(srcC)
        for i in 1:nF
            dCᵢ = ttm(C, FY[i].VᵀFV, 1)
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

# Root-to-leaves truncation of the augmented TTN.
# W is the weight of the node's mode 0, i.e. the singular values that the parent assigned to the
# basis functions of this subtree (W = 1 at the root). Without this weighting, the truncation of an
# inner node treats all of its basis functions as equally important, including directions with
# negligible weight at the parent, which led to large truncation errors.
function θ(obj::BUGIntegrator, Y::TTN, W::AbstractMatrix=ones(1, 1))
    nL = length(Y.leaves)
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;
    Cw = ttm(Y.C, Matrix(W), 1)
    Pfull = Matrix{Float64}[]
    Σfull = Vector{Float64}[]
    rnew = zeros(Int, nL)
    for i in 1:nL
        Pᵢ, Σ, _ = svd(tenmat(Cw, i+1))
        tol = obj.settings.ϵ * norm(Σ);

        # smallest rank such that the (ℓ¹-)tail of the discarded singular values is below tol;
        # if the tolerance cannot be met, all augmented directions are kept (capped by rMax)
        rmax = length(Σ)
        for j = 1:length(Σ)
            if sum(Σ[j:end]) < tol
                rmax = j;
                break;
            end
        end

        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,rMinTotal);
        push!(Pfull, Pᵢ); push!(Σfull, Σ); rnew[i] = min(rmax, size(Pᵢ, 2));
    end

    # rank consistency: rᵢ ≤ r₀ ∏_{j≠i} rⱼ (otherwise Matᵢ(C) cannot have rᵢ orthonormal rows,
    # which breaks the QR in prolong_and_retract_full). Can be violated when rMin is enforced.
    for _ in 1:nL, i in 1:nL
        rnew[i] = min(rnew[i], size(Y.C, 1) * prod(rnew[[j for j in 1:nL if j != i]]))
    end

    P = Array{Float64, 2}[]
    for i in 1:nL
        Pᵢ = Pfull[i][:, 1:rnew[i]]
        push!(P, Pᵢ')

        if length(Y.leaves[i].leaves) == 0 # Yᵢ is leaf
            Y.leaves[i].C = Y.leaves[i].C*Pᵢ
        else
            Yᵢ = copy_subtree(Y.leaves[i])
            Yᵢ.C = ttm(Yᵢ.C, Pᵢ', 1)
            Y.leaves[i] = θ(obj, Yᵢ, Diagonal(Σfull[i][1:rnew[i]]))
        end
    end
    Y.C = ttm(Y.C, P, [i+1 for i in 1:length(Y.leaves)])
    return Y
end

# rank adaptive BUG integrator
function SolveConventional(obj::BUGIntegrator)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;
    r = obj.settings.r;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN; # number PN moments
    nx = obj.settings.NCells; # number spatial cells

    # Set up initial condition
    u = zeros(nx,N);
    u[:,1] = IC(obj.settings.xMid)
    u[:,2] = IC(obj.settings.xMid)

    # truncate IC to rank r
    X,S,W = svd(u)
    X = X[:,1:r];
    S = diagm(S[1:r]);
    W = W[:,1:r];

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ= Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ= Diagonal(ones(nx)).*obj.settings.σₐ;

    # flux matrix and Roe matrix
    A = obj.rhs.A[1][2]';
    AbsA = obj.rhs.A[2][2]';
    Dₓ = obj.rhs.A[1][1]';
    Dₓₓ = obj.rhs.A[2][1]';

    rVec = zeros(2,nt)

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        r = size(S,1);

        # K-step
        K = X*S;
        #writedlm("K-value-conv.txt", K)
        WAW = W'*A'*W;
        WAbsAW = W'*AbsA'*W;
        WGW = W'*G*W;
        K = K .- Δt * Dₓ*K*WAW .+ Δt * Dₓₓ*K*WAbsAW .- Δt * σₐ*K .- Δt * σₛ*K*WGW; # advance K
        #println("conventional: ", - Dₓ*K*WAW .+ Dₓₓ*K*WAbsAW .- σₐ*K .- σₛ*K*WGW)
        #writedlm("K-rhs-conv.txt", - Dₓ*K*WAW .+ Dₓₓ*K*WAbsAW .- σₐ*K .- σₛ*K*WGW)
        X1,_ = np.linalg.qr([K X]); 
        M = X1'*X;

        # L-step
        XDₓₓX = X'*Dₓₓ*X;
        XDₓX = X'*Dₓ*X;
        XσₐX = X'*σₐ*X;
        XσₛX = X'*σₛ*X;
        L = W*S';
        #writedlm("L-value-conv.txt", L)
        L = L .- Δt * A*L*XDₓX' .+ Δt * AbsA*L*XDₓₓX' .- Δt * L*XσₐX .- Δt *G'*L*XσₛX; # advance L
        #println("conventional: ",  -A*L*XDₓX' .+ AbsA*L*XDₓₓX' .- L*XσₐX .- G'*L*XσₛX)
        #writedlm("L-rhs-conv.txt", -A*L*XDₓX' .+ AbsA*L*XDₓₓX' .- L*XσₐX .- G'*L*XσₛX)
        W1,_ = np.linalg.qr([L W]); 
        N = W1'*W;

        # S-step
        S = M*S*N';
        XDₓₓX = X1'*Dₓₓ*X1;
        XDₓX = X1'*Dₓ*X1;
        XσₐX = X1'*σₐ*X1;
        XσₛX = X1'*σₛ*X1;
        WAW = W1'*A'*W1;
        WAbsAW = W1'*AbsA'*W1;
        WGW = W1'*G*W1;
        S = S .- Δt * XDₓX*S*WAW .+ Δt * XDₓₓX*S*WAbsAW .- Δt * XσₐX*S .- Δt * XσₛX*S*WGW; # advance S

        # truncate
        X, S, W = truncate!(obj,X1,S,W1);

        rVec[1,n] = t;
        rVec[2,n] = r;

        t += Δt;

        #break;

        next!(prog) # update progress bar
    end

    # return end time and solution
    return X*S*W',rVec;
end

function truncate!(obj::BUGIntegrator,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2})
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(S);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;

    tmp = 0.0;
    tol = obj.settings.ϵ * norm(D);

    rmax = Int(floor(size(D,1)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
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

    # return rank
    return X*U[:, 1:rmax], diagm(D[1:rmax]), W*V[:, 1:rmax];
end
