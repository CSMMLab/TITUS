__precompile__

include("Problem.jl")
include("TTN.jl")

using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
np = pyimport("numpy")

struct Solver
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

function SetupIC(obj::Solver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.γ[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

function Solve(obj::Solver,xi::Float64=0.0)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = SetupIC(obj);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ .+ Diagonal(ones(nx)).* obj.σₛξ .* xi;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    A = obj.A;
    AbsA = obj.AbsA;

    # define inverse density
    s = obj.settings;

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        u = u .- Δt * obj.Dₓ*u*A' .+ Δt * obj.Dₓₓ*u*AbsA' .- Δt * σₐ*u .- Δt * σₛ*u*G; 
        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*u;

end

function F(obj::Solver, Y::Array{Float64,3})
    rhs = zeros(size(Y));
    d = 3;
    for i = 1:length(obj.Rhs)
        rhs .+= ttm(Y,Matrix.(obj.Rhs[i]),collect(1:d))
    end
    #return -ttm(Y, [Matrix(obj.Dₓ), Matrix(obj.A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])
    return rhs;
end

function precomputeProjection(obj::Solver, U⁰::Vector{Matrix{Float64}})
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

function F(obj::Solver, i::Int, r::Vector{Int}, K⁰::Matrix{Float64}, U⁰::Vector{Matrix{Float64}}, Q, rhsProject::Vector{Vector{Matrix{Float64}}}=[], precomputed=false)
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

function F(obj::Solver, U⁰::Vector{Matrix{Float64}}, C, rhsProject::Vector{Vector{Matrix{Float64}}}=[], precomputed=false)
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
    ¬ᵢ = collect(1:d); deleteat!(¬ᵢ, i);
    rᵢ = size(U⁰[i],2) 
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    V = tenmat(ttm(matten(Qᵀ',i,r), U⁰[¬ᵢ], ¬ᵢ), i)'
    K⁰ = U⁰[i]*S⁰;
    K¹ = K⁰ + Δt*Fᵢ(matten(K⁰*V', i, N))*V;
    Û¹,_ = np.linalg.qr([K¹ U⁰[i]], mode="reduced"); Û¹ = Matrix(Û¹[:,1:2*rᵢ]);
    return Û¹, Û¹'*U⁰[i]
end

# augment and update core tensor
function Ψ(C⁰, Û¹::Vector{Matrix{Float64}}, M::Vector{Matrix{Float64}}, F::Function, Δt, d::Int)
    Ĉ⁰ = ttm(C⁰, M, collect(1:d))
    return Ĉ⁰ + Δt * ttm(F(ttm(Ĉ⁰, Û¹, collect(1:d))), Matrix.(transpose.(Û¹)), collect(1:d))
end

function TuckerIntegratorStep(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, F::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    Û¹ = Matrix{Float64}[];
    M = Matrix{Float64}[];
    for i = 1:d
        Ûᵢ, Mᵢ = Φ(i, C⁰, U⁰, Y -> tenmat(F(Y), i), Δt, d, r, N)
        push!(Û¹, Ûᵢ); push!(M, Mᵢ);
    end
    Ĉ¹ = Ψ(C⁰, Û¹, M, Y -> F(Y), Δt, d);
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function TuckerIntegrator(obj::Solver)
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

        C⁰, U⁰, r = TuckerIntegratorStep(obj, C⁰, U⁰, Y -> F(obj, Y), Δt, 3, r, N)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

# update and augment the ith basis matrix
function ΦEfficient(i::Int, C⁰, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    rᵢ = size(U⁰[i],2) 
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    K⁰ = U⁰[i]*S⁰;
    K¹ = K⁰ + Δt*Fᵢ(K⁰,Qᵀ');
    Û¹,_ = np.linalg.qr([U⁰[i] K¹], mode="reduced"); Û¹ = Matrix(Û¹[:,1:2*rᵢ]);
    return Û¹, Û¹'*U⁰[i]
end

# update and augment the ith basis matrix
function pre_augment(i::Int, C⁰, U⁰::Vector{Matrix{Float64}}, Fᵢ::Function, Δt, d::Int, r::Vector{Int}, N::Vector{Int})
    rᵢ = size(U⁰[i],2)
    ¬ᵢ = collect(1:d); deleteat!(¬ᵢ, i);
    # augment core tensor
    for j in ¬ᵢ
        C⁰ = ttm(C⁰,[Matrix([diagm(ones(rᵢ)) zeros(rᵢ,rᵢ)]')],[j])
    end
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(C⁰, i)', mode="reduced"); S⁰ = Sᵀ'; 
    K⁰ = U⁰[i]*S⁰;
    Û⁰,_ = np.linalg.qr([U⁰[i] Fᵢ(K⁰,Qᵀ')], mode="reduced"); Û⁰ = Matrix(Û⁰[:,1:2*rᵢ]);
    return Û⁰, Û⁰'*U⁰[i]
end

# augment and update core tensor
function ΨEfficient(C⁰, M::Vector{Matrix{Float64}}, F::Function, Δt, dimRange::Vector{Int})
    Ĉ⁰ = ttm(C⁰, M, dimRange)
    return Ĉ⁰ + Δt * F(Ĉ⁰)
end

# augment and update core tensor
function ΨParallel(C⁰, F::Function, Δt)
    return Δt * F(C⁰)
end

function TuckerIntegratorStepEfficient(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute=true)
    Û¹ = Matrix{Float64}[];
    M = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
    end

    for i = 1:d
        Ûᵢ, Mᵢ = ΦEfficient(i, C⁰, U⁰, (K,Q) -> F(obj,i,r,K,U⁰,Q,rhsProject,precompute), Δt, d, r, N)
        push!(Û¹, Ûᵢ); push!(M, Mᵢ);
    end
    Ĉ¹ = ΨEfficient(C⁰, M, C -> F(obj, Û¹, C, Vector{Matrix{Float64}}[], false), Δt, collect(1:d));
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function ParallelTuckerIntegratorStep(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute=true)
    Û¹ = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
    end

    rField = []
    for i = 1:d
        push!(rField,1:r[i])
    end 

    # create tensor with double dimension
    Ĉ¹ = matten(zeros(Float64, 2*r[1], prod(2 .* r[2:end])),1,2 .* r)

    for i = 1:d
        Ûᵢ, _ = ΦEfficient(i, C⁰, U⁰, (K,Q) -> F(obj,i,r,K,U⁰,Q,rhsProject,precompute), Δt, d, r, N)

        rhsᵢ = deepcopy(rhsProject)
        Ũᵢ = Ûᵢ[:,(r[i]+1):end];
        for l in eachindex(rhsᵢ)
            rhsᵢ[l][i] = Ũᵢ'*obj.Rhs[l][i]*U⁰[i]
        end

        rFieldᵢ = deepcopy(rField); rFieldᵢ[i] = rField[i] .+ r[i]
        Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = ΨParallel(C⁰, C -> F(obj, U⁰, C, rhsᵢ, true), Δt);
        push!(Û¹, [U⁰[i] Ũᵢ]);
    end

    # precompute projections
    if !precompute
        rhsProject = Vector{Matrix{Float64}}[]
    end

    Ĉ¹[CartesianIndices(Tuple(rField))] = ΨEfficient(C⁰,Matrix{Float64}[],C -> F(obj, U⁰, C, rhsProject, precompute), Δt, Int[]);
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function ParallelTuckerIntegratorStep2ndOrder(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute=true)
    Û¹ = Matrix{Float64}[];
    Û⁰ = Matrix{Float64}[];

    # precompute projections
    if precompute
        rhsProject = precomputeProjection(obj, U⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
    end

    # augment basis
    for i in 1:d
        Û⁰[i], _ = pre_augment(i, C⁰, U⁰, (K,Q) -> F(obj,i,2*r,K,U⁰,Q,rhsProject,precompute), Δt, d, r, N)
    end

    # augment core tensor
    Ĉ⁰ = C⁰
    for j in 1:d
        Ĉ⁰ = ttm(Ĉ⁰,[Matrix([diagm(ones(rᵢ)) zeros(r[i],r[i])]')],[j])
    end

    # precompute projections
    if precompute
        rhsProject = precomputeProjection(obj, Û⁰)
    else
        rhsProject = Vector{Matrix{Float64}}[]
    end

    rField = []
    for i = 1:d
        push!(rField,1:r[i])
    end 

    # create tensor with double dimension
    Ĉ¹ = matten(zeros(Float64, 4*r[1], prod(4 .* r[2:end])),1,4 .* r)

    for i = 1:d
        Ûᵢ, _ = ΦEfficient(i, Ĉ⁰, Û⁰, (K,Q) -> F(obj,i,r,K,Û⁰,Q,rhsProject,precompute), Δt, d, r, N)

        rhsᵢ = deepcopy(rhsProject)
        Ũᵢ = Ûᵢ[:,(2*r[i]+1):end];
        for l in eachindex(rhsᵢ)
            rhsᵢ[l][i] = Ũᵢ'*obj.Rhs[l][i]*Û⁰[i]
        end

        rFieldᵢ = deepcopy(rField); rFieldᵢ[i] = rField[i] .+ r[i]
        Ĉ¹[CartesianIndices(Tuple(rFieldᵢ))] = ΨParallel(Ĉ⁰, C -> F(obj, Û⁰, C, rhsᵢ, true), Δt);
        push!(Û¹, [Û⁰[i] Ũᵢ]);
    end

    # precompute projections
    if !precompute
        rhsProject = Vector{Matrix{Float64}}[]
    end

    Ĉ¹[CartesianIndices(Tuple(rField))] = ΨEfficient(Ĉ⁰,Matrix{Float64}[],C -> F(obj, Û⁰, C, rhsProject, precompute), Δt, Int[]);
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 4 .* r);
    return Ĉ¹, Û¹, r
end

function TuckerIntegratorEfficient(obj::Solver)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    X = rand(s.NCells,r[1])
    V = rand(s.nPN,r[2])
    W = rand(s.Nxi,r[3])

    X[:,1] = 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);
    V[1,1] = 1; V[2:end,1] .= zeros(s.nPN-1);
    W[:,1] .= 1;
    C⁰ = zeros(s.r,s.r,s.r); C⁰[1,1,1] = 1;

    X, Rₓ = np.linalg.qr(X, mode="reduced");
    V, Rᵥ = np.linalg.qr(V, mode="reduced");
    W, R = np.linalg.qr(W, mode="reduced");
    C⁰ = ttm(C⁰,[Rₓ,Rᵥ,R],[1, 2, 3])
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = TuckerIntegratorStepEfficient(obj, C⁰, U⁰, Δt, 3, r, N)
        
        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function TTNIntegratorStep(obj::Solver, C⁰, U⁰::Vector{Matrix{Float64}}, Δt, d::Int, r::Vector{Int}, N::Vector{Int}, precompute=true)
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
    Ĉ¹ = ΨEfficient(C⁰, M, C -> F(obj, Û¹, C, Vector{Matrix{Float64}}[], false), Δt, collect(1:d));
    Ĉ¹, Û¹, r = θ(obj, Ĉ¹, Û¹, d, 2 .* r);
    return Ĉ¹, Û¹, r
end

function ParallelTuckerIntegrator(obj::Solver)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    X = rand(s.NCells,r[1])
    V = rand(s.nPN,r[2])
    W = rand(s.Nxi,r[3])

    X[:,1] = 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);
    V[1,1] = 1; V[2:end,1] .= zeros(s.nPN-1);
    W[:,1] .= 1;
    C⁰ = zeros(s.r,s.r,s.r); C⁰[1,1,1] = 1;

    X, Rₓ = np.linalg.qr(X, mode="reduced");
    V, Rᵥ = np.linalg.qr(V, mode="reduced");
    W, R = np.linalg.qr(W, mode="reduced");
    C⁰ = ttm(C⁰,[Rₓ,Rᵥ,R],[1, 2, 3])
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = ParallelTuckerIntegratorStep(obj, C⁰, U⁰, Δt, 3, r, N)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function ParallelTuckerIntegrator2ndOrder(obj::Solver)
    s = obj.settings;
    r = [s.r,s.r,s.r]

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = [s.NCells,s.nPN,s.Nxi];

    # Set up initial condition
    X = rand(s.NCells,r[1])
    V = rand(s.nPN,r[2])
    W = rand(s.Nxi,r[3])

    X[:,1] = 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);
    V[1,1] = 1; V[2:end,1] .= zeros(s.nPN-1);
    W[:,1] .= 1;
    C⁰ = zeros(s.r,s.r,s.r); C⁰[1,1,1] = 1;

    X, Rₓ = np.linalg.qr(X, mode="reduced");
    V, Rᵥ = np.linalg.qr(V, mode="reduced");
    W, R = np.linalg.qr(W, mode="reduced");
    C⁰ = ttm(C⁰,[Rₓ,Rᵥ,R],[1, 2, 3])
    U⁰ = [X, V, W];

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= r;

        C⁰, U⁰, r = ParallelTuckerIntegratorStep2ndOrder(obj, C⁰, U⁰, Δt, 3, r, N)

        t += Δt;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C⁰,U⁰,[1,2,3]), rankInTime;
end

function SolveBUG(obj::Solver)
    s = obj.settings;
    r = s.r;
    rVec = [r,r,r]
    debug = false

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=[r,r,r]);
    C = TT.cten; C = FillTensor(C,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    V = TT.fmat[2]; V = FillMatrix(V,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    
    A = obj.A;
    AbsA = obj.AbsA;

    ξ, w = gausslegendre(s.Nxi);
    σₛξ = Diagonal(obj.σₛξ .* ξ);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        
        ################## K1-step ##################
        QT,ST = np.linalg.qr(tenmat(C,1)', mode="reduced"); # decompose core tensor
        QMX = Matrix(QT);
        QX = matten(Matrix(QT)',1,rVec); S = Matrix(ST)';
        KX = X*S;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        rhsK = - obj.Dₓ*KX * (tenmat(ttm(QX,[V'*A*V],[2]),1) * QMX) .+ obj.Dₓₓ*KX * (tenmat(ttm(QX,[V'*AbsA*V],[2]),1) * QMX)
        rhsK .+= - σₐ*KX * (tenmat(QX,1) * QMX) .- σₛ*KX * (tenmat(ttm(QX,[V'*G'*V],[2]),1) * QMX) .- KX * (tenmat(ttm(QX,[V'*G'*V,U'* σₛξ * U],[2,3]),1) * QMX)

        KX = KX .+ Δt*rhsK;

        XNew,_ = np.linalg.qr(KX, mode="reduced");
        XNew = Matrix(XNew); S = Matrix(S);
        XNew = XNew[:,1:rVec[1]];
        MX = XNew'*X;

        # debug
        if debug
            i = 1
            Vtest = tenmat(ttm(matten(QT',1,rVec), [V, U], [2, 3]), i)'
            K¹ = X*S + Δt*tenmat(F(obj,matten(X*S*Vtest', i, [nx, N, s.Nxi])),i)*Vtest;
            println("X: ",norm(K¹ - KX))
        end

        ################## K2-step ##################
        QT,ST = np.linalg.qr(tenmat(C,2)', mode="reduced"); # decompose core tensor
        QMV = Matrix(QT);
        QV = matten(Matrix(QT)',2,rVec); S = Matrix(ST)';
        KV = V*S;

        rhsK = - A*KV * (tenmat(ttm(QV,[X'*obj.Dₓ*X],[1]),2) * QMV) .+ AbsA*KV * (tenmat(ttm(QV,[X'*obj.Dₓₓ*X],[1]),2) * QMV)
        rhsK .+= - KV * (tenmat(ttm(QV,[X'*σₐ*X],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[X'*σₛ*X],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[U'* σₛξ * U],[3]),2) * QMV)
        KV = KV .+ Δt * rhsK;

        VNew,_ = np.linalg.qr(KV, mode="reduced");
        VNew = Matrix(VNew); S = Matrix(S);
        VNew = VNew[:,1:rVec[2]];
        MW = VNew'*V;

        # debug
        if debug
            i = 2
            Vtest = tenmat(ttm(matten(QT',i,rVec), [X, U], [1, 3]), i)'
            K¹ = V*S + Δt*tenmat(F(obj,matten(V*S*Vtest', i, [nx, N, s.Nxi])),i)*Vtest;
            println("V: ",norm(K¹ - KV))
        end

        #######################################
        #Y = matten(V*S*Vtest', i, [nx, N, s.Nxi]);
        ##ttmTerm = - 0.0*ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ 0.0*ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- 0.0*ttm(Y, [Matrix(obj.σₐ)], [1]) .- 0.0*ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- 1.0*ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])
        #ttmTerm = ttm(QV, [X, Matrix(obj.G)*V*S, Matrix(obj.σₛξ .* ξ)*U], [1,2,3])
        #term1 = tenmat(ttmTerm,i)*Vtest;
        ##term2 = - 0.0*A*V*S * (tenmat(ttm(QV,[X'*obj.Dₓ*X],[1]),2) * QMV) .+ 0.0*AbsA*V*S * (tenmat(ttm(QV,[X'*obj.Dₓₓ*X],[1]),2) * QMV) - 0.0*V*S * (tenmat(ttm(QV,[X'*σₐ*X],[1]),2) * QMV) .- 0.0*G'*V*S * (tenmat(ttm(QV,[X'*σₛ*X],[1]),2) * QMV) .- 1.0*G'*V*S * (tenmat(ttm(QV,[U'* σₛξ * U],[3]),2) * QMV)
        ##term2 =  (tenmat(ttm(QV,[X'*X, G*V*S, U'* σₛξ * U],[1, 2, 3]),2) * QMV);
        #term2 =  tenmat(ttm(QV,[X, G*V*S, σₛξ * U],[1, 2, 3]),i) * Vtest;
        #println("debug ",norm(term1-term2))
        #######################################
        #-ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])

        ################## K3-step ##################
        QT,ST = np.linalg.qr(tenmat(C,3)', mode="reduced"); # decompose core tensor
        QMU = Matrix(QT);
        QU = matten(Matrix(QT)',3,rVec); S = Matrix(ST)';
        KU = U*S;

        rhsK = - KU * (tenmat(ttm(QU,[X'*obj.Dₓ*X,V'*A*V],[1,2]),3) * QMU) .+ KU * (tenmat(ttm(QU,[X'*obj.Dₓₓ*X,V'*AbsA*V],[1,2]),3) * QMU);
        rhsK .+= - KU * (tenmat(ttm(QU,[X'*σₐ*X],[1]),3) * QMU) .- KU * (tenmat(ttm(QU,[X'*σₛ*X,V'*G'*V],[1,2]),3) * QMU) .- σₛξ * KU * (tenmat(ttm(QU,[V'*G'*V],[2]),3) * QMU)
        KU = U*S .+ Δt*rhsK;

        UNew,_ = np.linalg.qr(KU, mode="reduced");
        UNew = Matrix(UNew); S = Matrix(S);
        UNew = UNew[:,1:rVec[3]];
        MU = UNew'*U;

        if debug
            i = 3
            Vtest = tenmat(ttm(matten(QT',i,rVec), [X, V], [1, 2]), i)' 
            K¹ = U*S + Δt*tenmat(F(obj,matten(U*S*Vtest', i, [nx, N, s.Nxi])),i)*Vtest;
            println("U: ",norm(K¹ - KU))
        end

        #######################################
        #Y = matten(U*S*Vtest', i, [nx, N, s.Nxi]);
        #ttmTerm = -ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])
        #term1 = tenmat(ttmTerm,i)*Vtest;
        #term2 = - U*S * (tenmat(ttm(QU,[X'*obj.Dₓ*X,V'*A*V],[1,2]),3) * QMU) .+ U*S * (tenmat(ttm(QU,[X'*obj.Dₓₓ*X,V'*AbsA*V],[1,2]),3) * QMU)- U*S * (tenmat(ttm(QU,[X'*σₐ*X],[1]),3) * QMU) .- U*S * (tenmat(ttm(QU,[X'*σₛ*X,V'*G'*V],[1,2]),3) * QMU) .- σₛξ * U*S * (tenmat(ttm(QU,[V'*G'*V],[2]),3) * QMU)
        #norm(term1-term2)
        #######################################

        #-ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])

        ################## C-step ##################
        X .= XNew;
        V .= VNew;
        U .= UNew;

        C = ttm(C,[MX,MW,MU],[1,2,3]);

        rhsC = -ttm(C,[X'*obj.Dₓ*X,V'*A*V,U'*U],[1,2,3]) .+ ttm(C,[X'*obj.Dₓₓ*X,V'*AbsA*V,U'*U],[1,2,3]) .- ttm(C,[X'*σₐ*X,V'*V,U'*U],[1,2,3]) .- ttm(C,[X'*σₛ*X,V'*G'*V,U'* U],[1,2,3]) .- ttm(C,[X'*X,V'*G'*V,U'* σₛξ * U],[1,2,3])
        C = C .+ Δt*rhsC;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C,[X,V,U],[1,2,3]);

end

function SolveBUGadaptive(obj::Solver)
    s = obj.settings;
    r = s.r;
    rVec = [r,r,r]
    debug = false

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=[r,r,r]);
    C = TT.cten; C = FillTensor(C,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    V = TT.fmat[2]; V = FillMatrix(V,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    
    A = obj.A;
    AbsA = obj.AbsA;

    σₛξ = obj.σₛξ;

    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= rVec;

        t += Δt;
        
        ################## K1-step ##################
        QT,ST = np.linalg.qr(tenmat(C,1)', mode="reduced"); # decompose core tensor
        QMX = Matrix(QT);
        QX = matten(Matrix(QT)',1,rVec); S = Matrix(ST)';
        KX = X*S;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        rhsK = - obj.Dₓ*KX * (tenmat(ttm(QX,[V'*A*V],[2]),1) * QMX) .+ obj.Dₓₓ*KX * (tenmat(ttm(QX,[V'*AbsA*V],[2]),1) * QMX)
        rhsK .+= - σₐ*KX * (tenmat(QX,1) * QMX) .- σₛ*KX * (tenmat(ttm(QX,[V'*G'*V],[2]),1) * QMX) .- KX * (tenmat(ttm(QX,[V'*G'*V,U'* σₛξ * U],[2,3]),1) * QMX)

        KX = KX .+ Δt*rhsK;

        XNew,_ = np.linalg.qr([KX X], mode="reduced");
        XNew = XNew[:,1:2*rVec[1]];
        MX = XNew'*X;

        # debug
        if debug
            i = 1
            Vtest = tenmat(ttm(matten(QT',1,rVec), [V, U], [2, 3]), i)'
            K¹ = X*S + Δt*tenmat(F(obj,matten(X*S*Vtest', i, [nx, N, s.Nxi])),i)*Vtest;
            println("KX: ",norm(K¹ - KX))
        end

        ################## K2-step ##################
        QT,ST = np.linalg.qr(tenmat(C,2)', mode="reduced"); # decompose core tensor
        QMV = Matrix(QT);
        QV = matten(Matrix(QT)',2,rVec); S = Matrix(ST)';
        KV = V*S;

        rhsK = - A*KV * (tenmat(ttm(QV,[X'*obj.Dₓ*X],[1]),2) * QMV) .+ AbsA*KV * (tenmat(ttm(QV,[X'*obj.Dₓₓ*X],[1]),2) * QMV)
        rhsK .+= - KV * (tenmat(ttm(QV,[X'*σₐ*X],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[X'*σₛ*X],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[U'* σₛξ * U],[3]),2) * QMV)
        KV = KV .+ Δt * rhsK;

        VNew,_ = np.linalg.qr([KV V], mode="reduced");
        VNew = Matrix(VNew); S = Matrix(S);
        VNew = VNew[:,1:2*rVec[2]];
        MW = VNew'*V;

        # debug
        if debug
            i = 2
            Vtest = tenmat(ttm(matten(QT',i,rVec), [X, U], [1, 3]), i)'
            K¹ = V*S + Δt*tenmat(F(obj,matten(V*S*Vtest', i, [nx, N, s.Nxi])),i)*Vtest;
            println("KV: ",norm(K¹ - KV))
        end

        #######################################
        #Y = matten(V*S*Vtest', i, [nx, N, s.Nxi]);
        ##ttmTerm = - 0.0*ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ 0.0*ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- 0.0*ttm(Y, [Matrix(obj.σₐ)], [1]) .- 0.0*ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- 1.0*ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])
        #ttmTerm = ttm(QV, [X, Matrix(obj.G)*V*S, Matrix(obj.σₛξ .* ξ)*U], [1,2,3])
        #term1 = tenmat(ttmTerm,i)*Vtest;
        ##term2 = - 0.0*A*V*S * (tenmat(ttm(QV,[X'*obj.Dₓ*X],[1]),2) * QMV) .+ 0.0*AbsA*V*S * (tenmat(ttm(QV,[X'*obj.Dₓₓ*X],[1]),2) * QMV) - 0.0*V*S * (tenmat(ttm(QV,[X'*σₐ*X],[1]),2) * QMV) .- 0.0*G'*V*S * (tenmat(ttm(QV,[X'*σₛ*X],[1]),2) * QMV) .- 1.0*G'*V*S * (tenmat(ttm(QV,[U'* σₛξ * U],[3]),2) * QMV)
        ##term2 =  (tenmat(ttm(QV,[X'*X, G*V*S, U'* σₛξ * U],[1, 2, 3]),2) * QMV);
        #term2 =  tenmat(ttm(QV,[X, G*V*S, σₛξ * U],[1, 2, 3]),i) * Vtest;
        #println("debug ",norm(term1-term2))
        #######################################
        #-ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])

        ################## K3-step ##################
        QT,ST = np.linalg.qr(tenmat(C,3)', mode="reduced"); # decompose core tensor
        QMU = Matrix(QT);
        QU = matten(Matrix(QT)',3,rVec); S = Matrix(ST)';
        KU = U*S;

        rhsK = - KU * (tenmat(ttm(QU,[X'*obj.Dₓ*X,V'*A*V],[1,2]),3) * QMU) .+ KU * (tenmat(ttm(QU,[X'*obj.Dₓₓ*X,V'*AbsA*V],[1,2]),3) * QMU);
        rhsK .+= - KU * (tenmat(ttm(QU,[X'*σₐ*X],[1]),3) * QMU) .- KU * (tenmat(ttm(QU,[X'*σₛ*X,V'*G'*V],[1,2]),3) * QMU) .- σₛξ * KU * (tenmat(ttm(QU,[V'*G'*V],[2]),3) * QMU)
        KU = U*S .+ Δt*rhsK;

        UNew,_ = np.linalg.qr([KU U], mode="reduced");
        UNew = Matrix(UNew); S = Matrix(S);
        UNew = UNew[:,1:2*rVec[3]];
        MU = UNew'*U;

        if debug
            i = 3
            Vtest = tenmat(ttm(matten(QT',i,rVec), [X, V], [1, 2]), i)' 
            K¹ = U*S + Δt*tenmat(F(obj,matten(U*S*Vtest', i, [nx, N, s.Nxi])),i)*Vtest;
            println("KU: ",norm(K¹ - KU))
        end

        #######################################
        #Y = matten(U*S*Vtest', i, [nx, N, s.Nxi]);
        #ttmTerm = -ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])
        #term1 = tenmat(ttmTerm,i)*Vtest;
        #term2 = - U*S * (tenmat(ttm(QU,[X'*obj.Dₓ*X,V'*A*V],[1,2]),3) * QMU) .+ U*S * (tenmat(ttm(QU,[X'*obj.Dₓₓ*X,V'*AbsA*V],[1,2]),3) * QMU)- U*S * (tenmat(ttm(QU,[X'*σₐ*X],[1]),3) * QMU) .- U*S * (tenmat(ttm(QU,[X'*σₛ*X,V'*G'*V],[1,2]),3) * QMU) .- σₛξ * U*S * (tenmat(ttm(QU,[V'*G'*V],[2]),3) * QMU)
        #norm(term1-term2)
        #######################################

        #-ttm(Y, [Matrix(obj.Dₓ), Matrix(A)], [1, 2]) .+ ttm(Y, [Matrix(obj.Dₓₓ), Matrix(obj.AbsA)], [1, 2]) .- ttm(Y, [Matrix(obj.σₐ)], [1]) .- ttm(Y, [Matrix(obj.σₛ), Matrix(obj.G)], [1,2]) .- ttm(Y, [Matrix(obj.G), Matrix(obj.σₛξ)], [2,3])

        if debug
            Û¹ = Matrix{Float64}[];
            M = Matrix{Float64}[];
            for i = 1:3
                Ûᵢ, Mᵢ = Φ(i, C, [X, V, U], Y -> tenmat(F(obj, Y), i), Δt, 3, rVec, [nx, N, s.Nxi])
                push!(Û¹, Ûᵢ); push!(M, Mᵢ);
            end
            println("error X: ", norm(XNew[1:2] - Û¹[1][1:2]))
            println("error V: ", norm(VNew[1:2] - Û¹[2][1:2]))
            println("error U: ", norm(UNew[1] - Û¹[3][1]))
        end

        ################## C-step ##################
        X = XNew;
        V = VNew;
        U = UNew;
        C0 = deepcopy(C)

        C = ttm(C,[MX,MW,MU],[1,2,3]);

        rhsC = -ttm(C,[X'*obj.Dₓ*X,V'*A*V,U'*U],[1,2,3]) .+ ttm(C,[X'*obj.Dₓₓ*X,V'*AbsA*V,U'*U],[1,2,3]) .- ttm(C,[X'*σₐ*X,V'*V,U'*U],[1,2,3]) .- ttm(C,[X'*σₛ*X,V'*G'*V,U'* U],[1,2,3]) .- ttm(C,[X'*X,V'*G'*V,U'* σₛξ * U],[1,2,3])
        C = C .+ Δt*rhsC;

        if debug
            Ĉ¹ = Ψ(C0, [X, V, U], [MX,MW,MU], Y -> F(obj, Y), Δt, 3);
            println("C: ",norm(Ĉ¹ - C))
        end

        C, Ufull, rVec = θ(obj, C, [X, V, U], 3, 2 .* rVec);
        X = Ufull[1]
        V = Ufull[2]
        U = Ufull[3]

        next!(prog) # update progress bar
    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C,[X,V,U],[1,2,3]), rankInTime;

end

function SolveParallel(obj::Solver)
    s = obj.settings;
    r = s.r;
    rVec = ones(Int, 3) * r;

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=[r,r,r]);
    C = TT.cten; C = FillTensor(C,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    V = TT.fmat[2]; V = FillMatrix(V,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    
    A = obj.A;
    AbsA = obj.AbsA;

    ξ, w = gausslegendre(s.Nxi);
    σₛξ = Diagonal(obj.σₛξ .* ξ);
    #XiWf = Diagonal(xi.*w.*0.5);
    #Wf = Diagonal(w.*0.5);
    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= rVec;
        
        ################## K1-step ##################
        QT,ST = np.linalg.qr(tenmat(C,1)', mode="reduced"); # decompose core tensor
        QMX = Matrix(QT);
        QX = matten(Matrix(QT)',1,rVec); S = Matrix(ST)';
        KX = X*S;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        rhsK = - obj.Dₓ*KX * (tenmat(ttm(QX,[V'*A*V],[2]),1) * QMX) .+ obj.Dₓₓ*KX * (tenmat(ttm(QX,[V'*AbsA*V],[2]),1) * QMX)
        rhsK .+= - σₐ*KX * (tenmat(QX,1) * QMX) .- σₛ*KX * (tenmat(ttm(QX,[V'*G'*V],[2]),1) * QMX) .- KX * (tenmat(ttm(QX,[V'*G'*V,U'* σₛξ * U],[2,3]),1) * QMX)

        KX = KX .+ Δt*rhsK;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        XNew,_ = np.linalg.qr([X KX], mode="reduced");
        X1Tilde = Matrix(XNew)[:, (rVec[1]+1):end];

        ################## K2-step ##################
        QT,ST = np.linalg.qr(tenmat(C,2)', mode="reduced"); # decompose core tensor
        QMV = Matrix(QT);
        QV = matten(Matrix(QT)',2,rVec); S = Matrix(ST)';
        KV = V*S;

        rhsK = - A*KV * (tenmat(ttm(QV,[X'*obj.Dₓ*X],[1]),2) * QMV) .+ AbsA*KV * (tenmat(ttm(QV,[X'*obj.Dₓₓ*X],[1]),2) * QMV)
        rhsK .+= - KV * (tenmat(ttm(QV,[X'*σₐ*X],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[X'*σₛ*X],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[U'* σₛξ * U],[3]),2) * QMV)
        KV = KV .+ Δt * rhsK;

        VNew,_ = np.linalg.qr([V KV], mode="reduced");
        V1Tilde = Matrix(VNew)[:, (rVec[2]+1):end];

        ################## K3-step ##################
        QT,ST = np.linalg.qr(tenmat(C,3)', mode="reduced"); # decompose core tensor
        QMU = Matrix(QT);
        QU = matten(Matrix(QT)',3,rVec); S = Matrix(ST)';
        KU = U*S;

        rhsK = - KU * (tenmat(ttm(QU,[X'*obj.Dₓ*X,V'*A*V],[1,2]),3) * QMU) .+ KU * (tenmat(ttm(QU,[X'*obj.Dₓₓ*X,V'*AbsA*V],[1,2]),3) * QMU);
        rhsK .+= - KU * (tenmat(ttm(QU,[X'*σₐ*X],[1]),3) * QMU) .- KU * (tenmat(ttm(QU,[X'*σₛ*X,V'*G'*V],[1,2]),3) * QMU) .- σₛξ * KU * (tenmat(ttm(QU,[V'*G'*V],[2]),3) * QMU)
        KU = KU .+ Δt*rhsK;

        UNew,_ = np.linalg.qr([U KU], mode="reduced");
        U1Tilde = Matrix(UNew)[:, (rVec[3]+1):end];

        ################## Cbar-step ##################

        rhsC = -ttm(C,[X'*obj.Dₓ*X,V'*A*V],[1,2]) .+ ttm(C,[X'*obj.Dₓₓ*X,V'*AbsA*V],[1,2]) .- ttm(C,[X'*σₐ*X],[1]) .- ttm(C,[X'*σₛ*X,V'*G'*V],[1,2]) .- ttm(C,[V'*G'*V,U'* σₛξ * U],[2,3])
        Cbar = C .+ Δt*rhsC;
        next!(prog) # update progress bar

        ################## augmentation-step ##################
        
        Chat = zeros(2*rVec[1],2*rVec[2],2*rVec[3]);
        Chat[1:rVec[1],1:rVec[2],1:rVec[3]] .= Cbar;
        Chat[(rVec[1]+1):end,1:rVec[2],1:rVec[3]] .= matten((X1Tilde'*KX) * tenmat(QX,1),1,rVec);
        Chat[1:rVec[1],(rVec[2]+1):end,1:rVec[3]] .= matten((V1Tilde'*KV) * tenmat(QV,2),2,rVec);
        Chat[1:rVec[1],1:rVec[2],(rVec[3]+1):end] .= matten((U1Tilde'*KU) * tenmat(QU,3),3,rVec);
        X1hat = [X X1Tilde]; V1hat = [V V1Tilde]; U1hat = [U U1Tilde];

        ################## truncation-step ##################
        rVec .= rVec .* 2;

        Ci = tenmat(Chat,1);
        X, Chat, rVec = truncate!(obj,X1hat,Ci,1,rVec)

        Ci = tenmat(Chat,2);
        V, Chat, rVec = truncate!(obj,V1hat,Ci,2,rVec)

        Ci = tenmat(Chat,3);
        U, C, rVec = truncate!(obj,U1hat,Ci,3,rVec)

        t += Δt;

    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C,[X,V,U],[1,2,3]), rankInTime;

end

# precomputes the r x r flux matrices
function SolveParallelPrecompute(obj::Solver)
    s = obj.settings;
    r = s.r;
    rVec = ones(Int, 3) * r;

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=[r,r,r]);
    C = TT.cten; C = FillTensor(C,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    V = TT.fmat[2]; V = FillMatrix(V,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    
    A = obj.A;
    AbsA = obj.AbsA;

    ξ, w = gausslegendre(s.Nxi);
    σₛξ = Diagonal(obj.σₛξ .* ξ);
    #XiWf = Diagonal(xi.*w.*0.5);
    #Wf = Diagonal(w.*0.5);
    rankInTime = zeros(4,nt);

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= rVec;

        # precompute flux matrices
        VᵀAV = V'*A*V;
        VᵀAbsAV = V'*AbsA*V;
        VᵀGV = V'*G'*V;
        UᵀσₛξU = U'* σₛξ * U;
        XᵀDₓX = X'*obj.Dₓ*X;
        XᵀDₓₓX = X'*obj.Dₓₓ*X;
        XᵀσₐX = X'*σₐ*X;
        XᵀσₛX = X'*σₛ*X

        ################## K1-step ##################
        QT,ST = np.linalg.qr(tenmat(C,1)', mode="reduced"); # decompose core tensor
        QMX = Matrix(QT);
        QX = matten(Matrix(QT)',1,rVec); S = Matrix(ST)';
        KX = X*S;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        rhsK = - obj.Dₓ*KX * (tenmat(ttm(QX,[VᵀAV],[2]),1) * QMX) .+ obj.Dₓₓ*KX * (tenmat(ttm(QX,[VᵀAbsAV],[2]),1) * QMX)
        rhsK .+= - σₐ*KX * (tenmat(QX,1) * QMX) .- σₛ*KX * (tenmat(ttm(QX,[VᵀGV],[2]),1) * QMX) .- KX * (tenmat(ttm(QX,[VᵀGV,UᵀσₛξU],[2,3]),1) * QMX)

        KX = KX .+ Δt*rhsK;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        XNew,_ = np.linalg.qr([X KX], mode="reduced");
        X1Tilde = Matrix(XNew)[:, (rVec[1]+1):end];

        ################## K2-step ##################
        QT,ST = np.linalg.qr(tenmat(C,2)', mode="reduced"); # decompose core tensor
        QMV = Matrix(QT);
        QV = matten(Matrix(QT)',2,rVec); S = Matrix(ST)';
        KV = V*S;

        rhsK = - A*KV * (tenmat(ttm(QV,[XᵀDₓX],[1]),2) * QMV) .+ AbsA*KV * (tenmat(ttm(QV,[XᵀDₓₓX],[1]),2) * QMV)
        rhsK .+= - KV * (tenmat(ttm(QV,[XᵀσₐX],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[XᵀσₛX],[1]),2) * QMV) .- G'*KV * (tenmat(ttm(QV,[UᵀσₛξU],[3]),2) * QMV)
        KV = KV .+ Δt * rhsK;

        VNew,_ = np.linalg.qr([V KV], mode="reduced");
        V1Tilde = Matrix(VNew)[:, (rVec[2]+1):end];

        ################## K3-step ##################
        QT,ST = np.linalg.qr(tenmat(C,3)', mode="reduced"); # decompose core tensor
        QMU = Matrix(QT);
        QU = matten(Matrix(QT)',3,rVec); S = Matrix(ST)';
        KU = U*S;

        rhsK = - KU * (tenmat(ttm(QU,[XᵀDₓX,VᵀAV],[1,2]),3) * QMU) .+ KU * (tenmat(ttm(QU,[XᵀDₓₓX,VᵀAbsAV],[1,2]),3) * QMU);
        rhsK .+= - KU * (tenmat(ttm(QU,[XᵀσₐX],[1]),3) * QMU) .- KU * (tenmat(ttm(QU,[XᵀσₛX,VᵀGV],[1,2]),3) * QMU) .- σₛξ * KU * (tenmat(ttm(QU,[VᵀGV],[2]),3) * QMU)
        KU = KU .+ Δt*rhsK;

        UNew,_ = np.linalg.qr([U KU], mode="reduced");
        U1Tilde = Matrix(UNew)[:, (rVec[3]+1):end];

        ################## Cbar-step ##################

        rhsC = -ttm(C,[XᵀDₓX,VᵀAV],[1,2]) .+ ttm(C,[XᵀDₓₓX,VᵀAbsAV],[1,2]) .- ttm(C,[XᵀσₐX],[1]) .- ttm(C,[XᵀσₛX,VᵀGV],[1,2]) .- ttm(C,[VᵀGV,UᵀσₛξU],[2,3])
        Cbar = C .+ Δt*rhsC;
        next!(prog) # update progress bar

        ################## augmentation-step ##################
        
        Chat = zeros(2*rVec[1],2*rVec[2],2*rVec[3]);
        Chat[1:rVec[1],1:rVec[2],1:rVec[3]] .= Cbar;
        Chat[(rVec[1]+1):end,1:rVec[2],1:rVec[3]] .= matten((X1Tilde'*KX) * tenmat(QX,1),1,rVec);
        Chat[1:rVec[1],(rVec[2]+1):end,1:rVec[3]] .= matten((V1Tilde'*KV) * tenmat(QV,2),2,rVec);
        Chat[1:rVec[1],1:rVec[2],(rVec[3]+1):end] .= matten((U1Tilde'*KU) * tenmat(QU,3),3,rVec);
        X1hat = [X X1Tilde]; V1hat = [V V1Tilde]; U1hat = [U U1Tilde];

        ################## truncation-step ##################
        rVec .= rVec .* 2;

        Ci = tenmat(Chat,1);
        X, Chat, rVec = truncate!(obj,X1hat,Ci,1,rVec)

        Ci = tenmat(Chat,2);
        V, Chat, rVec = truncate!(obj,V1hat,Ci,2,rVec)

        Ci = tenmat(Chat,3);
        U, C, rVec = truncate!(obj,U1hat,Ci,3,rVec)

        t += Δt;

    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C,[X,V,U],[1,2,3]),rankInTime;

end

function SolveParallelGeneral(obj::Solver)
    s = obj.settings;
    d = 3;
    r = s.r;
    rVec = ones(Int, d) * r;

    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = zeros(s.NCells,s.nPN,s.Nxi); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1,:] .= 2.0/sqrt(obj.γ[1])*IC(s,s.xMid);

    # obtain tensor representation
    TT = hosvd(u,reqrank=rVec);
    C = TT.cten;
    Y = [];
    for i = 1:d
        push!(Y,TT.fmat[i])
        rVec[i] = size(TT.fmat[i], 2);
    end
    push!(Y, TT.cten);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ=Diagonal(ones(nx)).*obj.settings.σₛ;
    σₐ=Diagonal(ones(nx)).*obj.settings.σₐ;
    
    A = obj.A;
    AbsA = obj.AbsA;

    ξ, w = gausslegendre(s.Nxi);
    σₛξ = Diagonal(obj.σₛξ .* ξ);
    rankInTime = zeros(4,nt);

    F1 = []; push!(F1,obj.Dₓ); push!(F1,A)

    prog = Progress(nt,1)
    #loop over time
    for n=1:nt

        rankInTime[1,n] = t;
        rankInTime[2:end,n] .= rVec;
        
        ################## K1-step ##################
        QT,ST = np.linalg.qr(tenmat(Y[end],1)', mode="reduced"); # decompose core tensor
        QK = matten(Matrix(QT)',1,rVec); S = Matrix(ST)';
        KX = Y[1]*S;

        #u = ttm(C,[X,V,U],[1,2,3]); # reconstruct solution
        #uC = ttm(Q,[K,V,U],[1,2,3]); # reconstruct solution with decomposed core tensor

        #rhsK = matten(

        rhsK = - ttm(QK,[obj.Dₓ*KX,Y[2]'*A*Y[2],Y[3]'*Y[3]],[1,2,3]) .+ ttm(QK,[obj.Dₓₓ*KX,V'*AbsA*V,Y[3]'*Y[3]],[1,2,3]) .- ttm(QK,[σₐ*KX,V'*V,Y[3]'*Y[3]],[1,2,3]) .- ttm(QK,[σₛ*KX,V'*G'*V,Y[3]'*Y[3]],[1,2,3]) .- ttm(QK,[KX,V'*G'*V,U'* σₛξ * U],[1,2,3])

        KX = KX .+ Δt*tenmat(rhsK,1)*tenmat(QK,1)';

        XNew,_ = np.linalg.qr([X KX], mode="reduced");
        X1Tilde = Matrix(XNew)[:, (rVec[1]+1):end];

        ################## K2-step ##################
        QT,ST = np.linalg.qr(tenmat(C,2)', mode="reduced"); # decompose core tensor
        QV = matten(Matrix(QT)',2,rVec); S = Matrix(ST)';
        KV = V*S;

        rhsK = - ttm(QV,[X'*obj.Dₓ*X,A*KV,U'*U],[1,2,3]) .+ ttm(QV,[X'*obj.Dₓₓ*X,AbsA*KV,U'*U],[1,2,3]) .- ttm(QV,[X'*σₐ*X,KV,U'*U],[1,2,3]) .- ttm(QV,[X'*σₛ*X,G'*KV,U' * U],[1,2,3]) .- ttm(QV,[X'*X,G'*KV,U'* σₛξ * U],[1,2,3])
        KV = KV .+ Δt*tenmat(rhsK,2)*tenmat(QV,2)';

        VNew,_ = np.linalg.qr([V KV], mode="reduced");
        V1Tilde = Matrix(VNew)[:, (rVec[2]+1):end];

        ################## K3-step ##################
        QT,ST = np.linalg.qr(tenmat(C,3)', mode="reduced"); # decompose core tensor
        QU = matten(Matrix(QT)',3,rVec); S = Matrix(ST)';
        KU = U*S;

        rhsK = - ttm(QU,[X'*obj.Dₓ*X,V'*A*V,KU],[1,2,3]) .+ ttm(QU,[X'*obj.Dₓₓ*X,V'*AbsA*V,KU],[1,2,3]) .- ttm(QU,[X'*σₐ*X,V'*V,KU],[1,2,3]) .- ttm(QU,[X'*σₛ*X,V'*G'*V,KU],[1,2,3]) .- ttm(QU,[X'*X,V'*G'*V,σₛξ * KU],[1,2,3])
        KU = KU .+ Δt*tenmat(rhsK,3)*tenmat(QU,3)';

        UNew,_ = np.linalg.qr([U KU], mode="reduced");
        U1Tilde = Matrix(UNew)[:, (rVec[3]+1):end];

        ################## Cbar-step ##################

        rhsC = -ttm(C,[X'*obj.Dₓ*X,V'*A*V,U'*U],[1,2,3]) .+ ttm(C,[X'*obj.Dₓₓ*X,V'*AbsA*V,U'*U],[1,2,3]) .- ttm(C,[X'*σₐ*X,V'*V,U'*U],[1,2,3]) .- ttm(C,[X'*σₛ*X,V'*G'*V,U'* U],[1,2,3]) .- ttm(C,[X'*X,V'*G'*V,U'* σₛξ * U],[1,2,3])
        Cbar = C .+ Δt*rhsC;
        next!(prog) # update progress bar

        ################## augmentation-step ##################
        
        Chat = zeros(2*rVec[1],2*rVec[2],2*rVec[3]);
        Chat[1:rVec[1],1:rVec[2],1:rVec[3]] .= Cbar;
        Chat[(rVec[1]+1):end,1:rVec[2],1:rVec[3]] .= matten((X1Tilde'*KX) * tenmat(QK,1),1,rVec);
        Chat[1:rVec[1],(rVec[2]+1):end,1:rVec[3]] .= matten((V1Tilde'*KV) * tenmat(QV,2),2,rVec);
        Chat[1:rVec[1],1:rVec[2],(rVec[3]+1):end] .= matten((U1Tilde'*KU) * tenmat(QU,3),3,rVec);
        X1hat = [X X1Tilde]; V1hat = [V V1Tilde]; U1hat = [U U1Tilde];

        for i = 1:d
            ind = [1:rVec[1],1:rVec[2],1:rVec[3]]
            Chat
        end

        ################## truncation-step ##################
        rVec .= rVec .* 2;

        Ci = tenmat(Chat,1);
        X, Chat, rVec = truncate!(obj,X1hat,Ci,1,rVec)

        Ci = tenmat(Chat,2);
        V, Chat, rVec = truncate!(obj,V1hat,Ci,2,rVec)

        Ci = tenmat(Chat,3);
        U, C, rVec = truncate!(obj,U1hat,Ci,3,rVec)

        t += Δt;

    end
    # return end time and solution
    return t, 0.5*sqrt(obj.γ[1])*ttm(C,[X,V,U],[1,2,3]),rankInTime;

end

function truncate!(obj::Solver,U::Array{Float64,2},Cᵢ::Array{Float64,2},i::Int,r::Vector{Int})
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

function θ(obj::Solver, Ĉ¹, Û¹::Vector{Matrix{Float64}}, d::Int, r::Vector{Int})
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
