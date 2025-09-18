__precompile__

include("Problem.jl")
include("TTN.jl")

using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
np = pyimport("numpy")

struct SolverTTN
    # spatial grid of cell interfaces
    x::Array{Float64};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    γ::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};
    # Roe matrix
    AbsA::Array{Float64,2};

    # stencil matrices
    Dₓ::Tridiagonal{Float64, Vector{Float64}};
    Dₓₓ::Tridiagonal{Float64, Vector{Float64}};

    # physical parameters
    σₐ::Diagonal{Float64, Vector{Float64}};
    σₛ::Diagonal{Float64, Vector{Float64}};
    σₛξ::Diagonal{Float64, Vector{Float64}};

    G::Diagonal{Float64, Vector{Float64}};

    Rhs

    # constructor
    function Solver(settings)
        x = settings.x;
        nx = settings.NCells;
        nξ = settings.Nxi;
        nΩ = settings.nPN
        Δx = settings.Δx;

        # setup flux matrix
        γ = ones(settings.nPN);

        # setup γ vector
        γ = zeros(settings.nPN);
        for i = 1:settings.nPN
            n = i-1;
            γ[i] = 2/(2*n+1);
        end
        
        # setup flux matrix
        A = zeros(settings.nPN,settings.nPN)

        for i = 1:(settings.nPN-1)
            n = i-1;
            A[i,i+1] = (n+1)/(2*n+1)*sqrt(γ[i+1])/sqrt(γ[i]);
        end

        for i = 2:settings.nPN
            n = i-1;
            A[i,i-1] = n/(2*n+1)*sqrt(γ[i-1])/sqrt(γ[i]);
        end

        # setup Roe matrix
        S = eigvals(A)
        V = eigvecs(A)
        AbsA = V*abs.(diagm(S))*inv(V)

        # set up spatial stencil matrices
        Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
        Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

        #Compute diagonal of scattering matrix G
        G = Diagonal([0.0;ones(settings.nPN-1)]);
        σₛ = Diagonal(ones(nx)).*settings.σₛ;
        σₐ = Diagonal(ones(nx)).*settings.σₐ;
        

        ξ, w = gausslegendre(settings.Nxi);
        σₛξ = Diagonal(settings.σₛξ .* ξ);

        # setup right hand side 
        Rhs = [];
        RhsTerm = [-Dₓ, A, Diagonal(ones(nξ))]
        push!(Rhs,RhsTerm)
        RhsTerm = [Dₓₓ, AbsA, Diagonal(ones(nξ))]
        push!(Rhs,RhsTerm)
        RhsTerm = [-σₐ, Diagonal(ones(nΩ)), Diagonal(ones(nξ))]
        push!(Rhs,RhsTerm)
        RhsTerm = [-σₛ, G, Diagonal(ones(nξ))]
        push!(Rhs,RhsTerm)
        RhsTerm = [Diagonal(ones(nx)), G, σₛξ]
        push!(Rhs,RhsTerm)

        new(x,settings,γ,A,AbsA,Dₓ,Dₓₓ,σₐ,σₛ,σₛξ, G, Rhs);
    end
end

function SetupIC(obj::SolverTTN)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.γ[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

function F(obj::SolverTTN, Y::Array{Float64,3})
    rhs = zeros(size(Y));
    d = 3;
    for i = 1:length(obj.Rhs)
        rhs .+= ttm(Y,Matrix.(obj.Rhs[i]),collect(1:d))
    end
    return rhs;
end

function precomputeProjection(obj::SolverTTN, U⁰::Vector{Matrix{Float64}})
    rhsProject = Vector{Matrix{Float64}}[]
    d = length(U⁰)
    for j = 1:length(obj.Rhs)
        termProjected = Matrix{Float64}[]
        for l in 1:d
            push!(termProjected, U⁰[l]'*obj.Rhs[j][l]*U⁰[l])
        end
        push!(rhsProject,termProjected)
    end
    return rhsProject
end

function F(obj::SolverTTN, i::Int, r::Vector{Int}, K⁰::Matrix{Float64}, U⁰::Vector{Matrix{Float64}}, Q, rhsProject::Vector{Vector{Matrix{Float64}}}=[], precomputed=false)
    d = 3;
    ¬ᵢ = collect(1:d); deleteat!(¬ᵢ, i);
    rhs = zeros(size(K⁰));
    Qtensor = matten(Q,i,r)
    
    if !precomputed
        for j = 1:length(obj.Rhs)
            RhsProjected = Matrix{Float64}[]
            for l in 1:d
                if l != i
                    push!(RhsProjected, U⁰[l]'*obj.Rhs[j][l]*U⁰[l])
                else
                    push!(RhsProjected, obj.Rhs[j][l]*K⁰)
                end
            end
            rhs .+= obj.Rhs[j][i]*K⁰*(tenmat(ttm(Qtensor, RhsProjected[¬ᵢ], ¬ᵢ), i)*Q')
        end
    else
        for j = 1:length(obj.Rhs)
            rhs .+= obj.Rhs[j][i]*K⁰*(tenmat(ttm(Qtensor, rhsProject[j][¬ᵢ], ¬ᵢ), i)*Q')
        end
    end
    return rhs
end

function F(obj::SolverTTN, U⁰::Vector{Matrix{Float64}}, C, rhsProject::Vector{Vector{Matrix{Float64}}}=[], precomputed=false)
    d = 3;
    rhs = zeros(size(C));
    
    if !precomputed
        for j = 1:length(obj.Rhs)
            RhsProjected = Matrix{Float64}[]
            for l in 1:d
                push!(RhsProjected, U⁰[l]'*obj.Rhs[j][l]*U⁰[l])
            end
            rhs .+= ttm(C, RhsProjected, collect(1:d))
        end
    else
        for j = 1:length(obj.Rhs)
            rhs .+= ttm(C, rhsProject[j], collect(1:d))
        end
    end
    return rhs
end

# update and augment the ith basis matrix
function Φ(i::Int, C⁰, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    rᵢ = size(U⁰[i],2) 
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    if i == is_leaf
        K⁰ = U⁰[i]*S⁰;
        K¹ = K⁰ + Δt*Fᵢ(K⁰,Qᵀ');
        Û¹,_ = np.linalg.qr([U⁰[i] K¹], mode="reduced"); Û¹ = Matrix(Û¹[:,1:2*rᵢ]);
        M = Û¹'*U⁰[i]
    else

    end
    
    return Û¹, M
end

# augment and update core tensor
function Ψ(C⁰, M::Vector{Matrix{Float64}}, F::Function, Δt, dimRange::Vector{Int})
    Ĉ⁰ = ttm(C⁰, M, dimRange)
    return Ĉ⁰, Ĉ⁰ + Δt * F(Ĉ⁰)
end

function Step(obj::SolverTTN, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute=true)
    Û¹ = Matrix{Float64}[];
    M = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
    end

    for i = 1:d
        Ûᵢ, Mᵢ = Φ(i, C⁰, U⁰, (K,Q) -> F(obj,i,r,K,U⁰,Q,rhsProject,precompute), Δt, d, r, N)
        push!(Û¹, Ûᵢ); push!(M, Mᵢ);
    end
    Ĉ⁰, Ĉ¹ = Ψ(C⁰, M, C -> F(obj, Û¹, C, Vector{Matrix{Float64}}[], false), Δt, collect(1:d));
    return Ĉ⁰, Ĉ¹, Û¹
end

function Solve(obj::SolverTTN)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=r);
    C⁰ = TT.cten; C⁰ = FillTensor(C⁰,r);
    X = TT.fmat[1]; X = FillMatrix(X,r[1]);
    V = TT.fmat[2]; V = FillMatrix(V,r[2]);
    W = TT.fmat[3]; W = FillMatrix(W,r[3]);
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        _, C⁰, U⁰, r = Step(obj, C⁰, U⁰, Y -> F(obj, Y), Δt, 3, r, N)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function truncate!(obj::SolverTTN,U::Array{Float64,2},Cᵢ::Array{Float64,2},i::Int,r::Vector{Int})
    # Compute singular values of S and decide how to truncate:
    P,D,Q = svd(Cᵢ);
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = obj.settings.rMin;

    tmp = 0.0;
    tol = obj.settings.ϵ;#*norm(D);

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

    r[i] = rmax;
    C = matten(Diagonal(D[1:rmax])*Q[:,1:rmax]',i,r);

    # return rank
    return U*P[:, 1:rmax], C, r;
end

function θ(obj::SolverTTN, Ĉ¹, Û¹::Vector{Matrix{Float64}}, d::Int, r::Vector{Int})
    for i = 1:d
        Cᵢ = tenmat(Ĉ¹,i);
        Û¹[i], Ĉ¹, r = truncate!(obj,Û¹[i],Cᵢ,i,r)
    end
    return Ĉ¹, Û¹, r
end

function FillTensor(ten,r::Int)
    if size(ten,1) != r || size(ten,2) != r || size(ten,3) != r
        tmp = ten;
        ten = zeros(r,r,r)
        for i = 1:size(tmp,1)
            for j = 1:size(tmp,2)
                for k = 1:size(tmp,3)
                    ten[i,j,k] = tmp[i,j,k];
                end
            end
        end
    end
    return ten
end

function FillTensor(ten, r::Vector{Int})
    if size(ten,1) != r[1] || size(ten,2) != r[2] || size(ten,3) != r[3]
        tmp = ten;
        ten = zeros(r[1],r[2],r[3])
        for i = 1:size(tmp,1)
            for j = 1:size(tmp,2)
                for k = 1:size(tmp,3)
                    ten[i,j,k] = tmp[i,j,k];
                end
            end
        end
    end
    return ten
end

function FillMatrix(mat,r)
    if size(mat,2) != r
        tmp = mat;
        mat = zeros(size(tmp,1),r)
        for i = 1:size(tmp,2)
            mat[:,i] = tmp[:,i];
        end
    end
    return mat
end
