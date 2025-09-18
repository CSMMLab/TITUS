using FastGaussQuadrature

struct Problem
    Rhs
    function Problem(settings::Settings)
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

        new(Rhs);
    end
end

# projected right hand side when input is factorized tensor. Has option to use precomputed projection of right-hand side
function F(obj::Problem, U⁰::Vector{Matrix{Float64}}, C, rhsProject::Vector{Vector{Matrix{Float64}}}=[], precomputed=false)
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

# projected right hand side except for mode i when input is factorized tensor. Has option to use precomputed projection of right-hand side
function F(obj::Problem, i::Int, r::Vector{Int}, K⁰::Matrix{Float64}, U⁰::Vector{Matrix{Float64}}, Q, rhsProject::Vector{Vector{Matrix{Float64}}}=[], precomputed=false)
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

#classical right hand side when input is tensor
function F(obj::Problem, Y::Array{Float64,3})
    rhs = zeros(size(Y));
    d = 3;
    for i = 1:length(obj.Rhs)
        rhs .+= ttm(Y,Matrix.(obj.Rhs[i]),collect(1:d))
    end
    return rhs;
end