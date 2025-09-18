using LinearAlgebra
include("stencil.jl")
include("PNSystem.jl")

# tree tensor network operator. Consists of several flux matrices at leaves of several TTNs
struct Rhs
    A::Vector{Array} # flux matrices for leaves
    wξ::Vector{Float64}
    wη::Vector{Float64}
    function Rhs(settings::Settings)
        problem_type = settings.problem
        if problem_type == "radiation"
            nx = settings.NCells;
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
            AbsA = V*abs.(Diagonal(S))*inv(V)

            # set up spatial stencil matrices
            Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
            Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

            #Compute diagonal of scattering matrix G
            G = Diagonal([0.0;ones(settings.nPN-1)]);
            σₛ = Diagonal(ones(nx)).*settings.σₛ;
            σₐ = Diagonal(ones(nx)).*settings.σₐ;

            # setup right hand side 
            ARhs = [];
            RhsTerm = [-Dₓ, A]
            push!(ARhs,RhsTerm)
            RhsTerm = [Dₓₓ, AbsA]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₐ, Diagonal(ones(nΩ))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₛ, G]
            push!(ARhs,RhsTerm)
            new(ARhs);
        elseif problem_type == "radiationUQ"
            nx = settings.NCells;
            nξ = settings.Nxi;
            nη = settings.Neta;
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
            AbsA = V*abs.(Diagonal(S))*inv(V)

            # set up spatial stencil matrices
            Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
            Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

            #Compute diagonal of scattering matrix G
            G = Diagonal([0.0;ones(settings.nPN-1)]);
            σₛ = Diagonal(ones(nx)).*settings.σₛ;
            σₐ = Diagonal(ones(nx)).*settings.σₐ;

            ξ, wξ = gausslegendre(nξ);
            η, wη = gausslegendre(nη);
            σₛξ = Diagonal(settings.σₛξ .* ξ);
            σₛη = Diagonal(settings.σₛη .* η);

            # setup right hand side 
            ARhs = [];
            RhsTerm = [-Dₓ, A, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [Dₓₓ, AbsA, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₐ, Diagonal(ones(nΩ)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₛ, G, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, σₛξ, Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, Diagonal(ones(nξ)), σₛη]
            push!(ARhs,RhsTerm)
            new(ARhs,wξ,wη);
        
        elseif problem_type == "radiationUQ8D"
            nx = settings.NCells;
            nξ = settings.Nxi;
            nη = settings.Neta;
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
            AbsA = V*abs.(Diagonal(S))*inv(V)

            # set up spatial stencil matrices
            Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
            Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

            #Compute diagonal of scattering matrix G
            G = Diagonal([0.0;ones(settings.nPN-1)]);
            σₛ = Diagonal(ones(nx)).*settings.σₛ;
            σₐ = Diagonal(ones(nx)).*settings.σₐ;

            ξ, wξ = gausslegendre(nξ);
            η, wη = gausslegendre(nη);
            σₛξ = Diagonal(settings.σₛξ .* ξ);
            σₛη = Diagonal(settings.σₛη .* η);
            σₛξ₁ = Diagonal(settings.σₛξ .* ξ.^2);
            σₛη₁ = Diagonal(settings.σₛη .* η.^2);
            σₛξ₂ = Diagonal(settings.σₛξ .* ξ.^3);
            σₛη₂ = Diagonal(settings.σₛη .* η.^3);

            # setup right hand side 
            ARhs = [];
            RhsTerm = [-Dₓ, A, Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [Dₓₓ, AbsA, Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₐ, Diagonal(ones(nΩ)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₛ, G, Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, σₛξ, Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, Diagonal(ones(nξ)), σₛη, Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, Diagonal(ones(nξ)), Diagonal(ones(nξ)), σₛξ₁, Diagonal(ones(nη)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, Diagonal(ones(nξ)), Diagonal(ones(nξ)), Diagonal(ones(nξ)), σₛη₁, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, Diagonal(ones(nξ)), Diagonal(ones(nξ)), Diagonal(ones(nξ)), Diagonal(ones(nη)), σₛξ₂, Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Diagonal(ones(nx)), G, Diagonal(ones(nξ)), Diagonal(ones(nξ)), Diagonal(ones(nξ)), Diagonal(ones(nη)), Diagonal(ones(nξ)), σₛη₂]
            push!(ARhs,RhsTerm)
            new(ARhs,wξ,wη);
        elseif problem_type == "radiation2DUQ" || problem_type == "Lattice"
            nx = settings.NCells
            ny = nx
            ncells = nx*ny;
            nξ = settings.Nxi;
            nη = settings.Neta;
            nΩ = GlobalIndex( s.nPN, s.nPN ) + 1
            Δx = settings.Δx;
            xMid = settings.xMid
            yMid = settings.xMid
            # construct PN system matrices
            pn = PNSystem(settings)
            SetupSystemMatrices(pn)

            # setup Roe matrix
            S = eigvals(pn.Ax)
            V = eigvecs(pn.Ax)
            AbsAx = V * abs.(diagm(S)) * inv(V)

            S = eigvals(pn.Az)
            V = eigvecs(pn.Az)
            AbsAz = V * abs.(diagm(S)) * inv(V)

            stencil = Stencil(settings, settings.NCells, settings.NCells)
            Dxx = stencil.Dxx
            Dx = stencil.Dx
            Dyy = stencil.Dyy
            Dy = stencil.Dy

            #Compute diagonal of scattering matrix G
            G = Diagonal([0.0;ones(nΩ-1)]);

            if problem_type == "radiation2DUQ"
                σₛ = settings.σₛ .* ones(nx * ny)
                σₐ = settings.σₐ .* ones(nx * ny)
            elseif problem_type == "Lattice"
                σₛ = ones(nx * ny)
                σₐ = zeros(nx * ny)
                Q = zeros(nx * ny)
                for i = 1:nx
                    for j = 1:ny
                        if (xMid[i] <= 2.0 && xMid[i] >= 1.0) || (xMid[i] <= 6.0 && xMid[i] >= 5.0)
                            if (yMid[j] <= 2.0 && yMid[j] >= 1.0) || (yMid[j] <= 4.0 && yMid[j] >= 3.0) || (yMid[j] <= 6.0 && yMid[j] >= 5.0)
                                σₛ[vectorIndex(ny, i, j)] = 0.0
                                σₐ[vectorIndex(ny, i, j)] = 10.0
                            end
                        end
                        if (xMid[i] <= 3.0 && xMid[i] >= 2.0) || (xMid[i] <= 5.0 && xMid[i] >= 4.0)
                            if (yMid[j] <= 3.0 && yMid[j] >= 2.0) || (yMid[j] <= 5.0 && yMid[j] >= 4.0)
                                σₛ[vectorIndex(ny, i, j)] = 0.0
                                σₐ[vectorIndex(ny, i, j)] = 10.0
                            end
                        end
                        if xMid[i] <= 4.0 && xMid[i] >= 3.0
                            if yMid[j] <= 6.0 && yMid[j] >= 5.0
                                σₛ[vectorIndex(ny, i, j)] = 0.0
                                σₐ[vectorIndex(ny, i, j)] = 10.0
                            elseif yMid[j] <= 4.0 && yMid[j] >= 3.0
                                σₛ[vectorIndex(ny, i, j)] = 0.0
                                σₐ[vectorIndex(ny, i, j)] = 10.0
                                Q[vectorIndex(ny, i, j)] = 1.0
                            end
                        end
                    end
                end
            end
            σₛ = Diagonal(σₛ)
            σₐ = Diagonal(σₐ)

            ξ, wξ = gausslegendre(nξ);
            η, wη = gausslegendre(nη);
            σₛξ = Diagonal(settings.σₛξ .* ξ);
            σₛη = Diagonal(settings.σₛη .* η);

            e₁ = Diagonal(zeros(nΩ))
            e₁[1, 1] = 1.0

            # setup right hand side 
            ARhs = [];
            QRhs = [];
            RhsTerm = [-Dx, pn.Ax, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Dxx, AbsAx, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Dy, pn.Az, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-Dyy, AbsAz, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₐ, Diagonal(ones(nΩ)), Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            RhsTerm = [-σₛ, G, Diagonal(ones(nξ)), Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            #RhsTerm = [-σₛ, G, σₛξ, Diagonal(ones(nη))]
            RhsTerm = [Diagonal(ones(ncells)), G, σₛξ, Diagonal(ones(nη))]
            push!(ARhs,RhsTerm)
            #RhsTerm = [-σₐ, Diagonal(ones(nΩ)), Diagonal(ones(nξ)), σₛη]
            RhsTerm = [Diagonal(ones(ncells)), Diagonal(ones(nΩ)), Diagonal(ones(nξ)), σₛη]
            push!(ARhs,RhsTerm)
            if problem_type == "Lattice"
                QRhs = generateSource(settings, Q)
            end
            new(ARhs, wξ, wη);
        elseif problem_type == "IsingModel"
            id = Diagonal(ones(2))
            σx = [0.0 1.0; 1.0 0.0]
            σz = [1.0 0.0; 0.0 -1.0]
            Ω = 1
            Dₓₓ = Ω * σx

            ARhs = [];
            RhsTerm = [Dₓₓ, id, id, id]
            push!(ARhs,RhsTerm)
            RhsTerm = [id, Dₓₓ, id, id]
            push!(ARhs,RhsTerm)
            RhsTerm = [id, id, Dₓₓ, id]
            push!(ARhs,RhsTerm)
            RhsTerm = [id, id, id, Dₓₓ]
            push!(ARhs,RhsTerm)
            RhsTerm = [σz, σz, id, id]
            push!(ARhs,RhsTerm)
            RhsTerm = [id, σz, σz, id]
            push!(ARhs,RhsTerm)
            RhsTerm = [id, id, σz, σz]
            push!(ARhs,RhsTerm)
            RhsTerm = [σz, id, id, σz]
            push!(ARhs,RhsTerm)
            new(ARhs);
        else
            nx = settings.NCells;
            id = Diagonal(ones(nx))
            Δx = settings.Δx;
            Dₓₓ = Tridiagonal(ones(nx-1)/2.0,-ones(nx),ones(nx-1)/2.0) # stabilization matrix

            ARhs = [];
            RhsTerm = [Dₓₓ, id, id, id]
            push!(ARhs,RhsTerm)
            RhsTerm = [id, Dₓₓ, id, id]
            push!(ARhs,RhsTerm)
            new(ARhs);
        end
    end
end

include("TTN.jl")

# evaluates right-hand side (Rhs) at Y, which is a list of TTNs
function eval(obj::Rhs, Y::TTN)
    FY = TTN[]
    leavesY = get_leaf_nodes(Y)
    idx = [node.id for node in leavesY ]
    for Aᵢ in obj.A
        FYᵢ = copy_subtree(Y)
        leaves = get_leaf_nodes(FYᵢ)
        for (Aᵢⱼ, leaf) in zip(Aᵢ[idx], leaves) # adapt leaves of FYᵢ by multiplying corresponding flux matrix
            leaf.C = Aᵢⱼ*leaf.C
        end
        push!(FY, FYᵢ)
    end
    return FY
end