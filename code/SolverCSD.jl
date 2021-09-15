__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays

include("CSD.jl")
include("PNSystem.jl")

struct SolverCSD
    # spatial grid of cell interfaces
    x::Array{Float64};
    y::Array{Float64};

    # Solver settings
    settings::Settings;

    # preallocate memory for performance
    outRhs::Array{Float64,3};
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # Roe matrix
    AbsAx::Array{Float64,2};
    AbsAz::Array{Float64,2};
    # normalized Legendre Polynomials
    P::Array{Float64,2};
    # quadrature points
    mu::Array{Float64,1};
    w::Array{Float64,1};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # material density
    density::Array{Float64,2};
    densityVec::Array{Float64,1};

    # dose vector
    dose::Array{Float64,1};

    L1x::SparseMatrixCSC{Float64, Int64};
    L1y::SparseMatrixCSC{Float64, Int64};
    L2x::SparseMatrixCSC{Float64, Int64};
    L2y::SparseMatrixCSC{Float64, Int64};

    # constructor
    function SolverCSD(settings)
        x = settings.x;
        y = settings.y;

        # setup flux matrix
        gamma = zeros(settings.nPN+1);
        for i = 1:settings.nPN+1
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        A = zeros(settings.nPN,settings.nPN);
            # setup flux matrix (alternative analytic computation)
        for i = 1:(settings.nPN-1)
            n = i-1;
            A[i,i+1] = (n+1)/(2*n+1)*sqrt(gamma[i+1])/sqrt(gamma[i]);
        end

        for i = 2:settings.nPN
            n = i-1;
            A[i,i-1] = n/(2*n+1)*sqrt(gamma[i-1])/sqrt(gamma[i]);
        end

        # construct CSD fields
        csd = CSD(settings);

        # construct PN system matrices
        pn = PNSystem(settings)
        SetupSystemMatrices(pn);

        outRhs = zeros(settings.NCellsX,settings.NCellsY,pn.nTotalEntries);

        # setup Roe matrix
        S = eigvals(pn.Ax)
        V = eigvecs(pn.Ax)
        AbsAx = V*abs.(diagm(S))*inv(V)

        S = eigvals(pn.Az)
        V = eigvecs(pn.Az)
        AbsAz = V*abs.(diagm(S))*inv(V)

        # set density vector
        density = settings.density;

        # allocate dose vector
        dose = zeros(settings.NCellsX*settings.NCellsY)

        # compute normalized Legendre Polynomials
        Nq=200;
        (mu,w) = gauss(Nq);
        P=zeros(Nq,settings.nPN);
        for k=1:Nq
            PCurrent = collectPl(mu[k],lmax=settings.nPN-1);
            for i = 1:settings.nPN
                P[k,i] = PCurrent[i-1]/sqrt(gamma[i]);
            end
        end

        # setupt stencil matrix
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        N = pn.nTotalEntries;
        L1x = spzeros(nx*ny,nx*ny);
        L1y = spzeros(nx*ny,nx*ny);
        L2x = spzeros(nx*ny,nx*ny);
        L2y = spzeros(nx*ny,nx*ny);
        for i = 2:nx-1
            for j = 2:ny-1
                # x part
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);
                L1x[index,index] = 2.0/2/settings.dx/density[i,j]; 
                if i > 1
                    L1x[index,indexMinus] = -1/2/settings.dx/density[i-1,j]; 
                    L2x[index,indexMinus] = -1/2/settings.dx/density[i-1,j]; 
                end
                if i < nx
                    L1x[index,indexPlus] = -1/2/settings.dx/density[i+1,j]; 
                    L2x[index,indexPlus] = 1/2/settings.dx/density[i+1,j];
                end

                # y part
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);
                L1y[index,index] += 2.0/2/settings.dy/density[i,j]; 
                if j > 1
                    L1y[index,indexMinus] = -1/2/settings.dy/density[i,j-1]; 
                    L2y[index,indexMinus] = -1/2/settings.dy/density[i,j-1];
                end
                if j < ny
                    L1y[index,indexPlus] = -1/2/settings.dy/density[i,j+1]; 
                    L2y[index,indexPlus] = 1/2/settings.dy/density[i,j+1];
                end
            end
        end

        new(x,y,settings,outRhs,gamma,AbsAx,AbsAz,P,mu,w,csd,pn,density,vec(density'),dose,L1x,L1y,L2x,L2y);
    end
end

function SetupIC(obj::SolverCSD)
    u = zeros(obj.settings.NCellsX,obj.settings.NCellsY,obj.pn.nTotalEntries);
    if obj.settings.problem == "2DDirected"
        PCurrent = collectPl(1,lmax=obj.settings.nPN);
        for l = 0:obj.settings.nPN
            for k=-l:l
                i = GlobalIndex( l, k )+1;
                u[:,:,i] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid) .* PCurrent[l]/sqrt(obj.gamma[l+1])
            end
        end
    elseif obj.settings.problem == "2D" || obj.settings.problem == "2DHighD"
        for l = 0:obj.settings.nPN
            for k=-l:l
                i = GlobalIndex( l, k )+1;
                u[:,:,i] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)*obj.csd.StarMAPmoments[i]
            end
        end
    end
    return u;
end

function PsiLeft(obj::SolverCSD,n::Int,mu::Float64)
    E0 = obj.settings.eMax;
    return 10^5*exp(-200.0*(1.0-mu)^2)*exp(-50*(E0-E)^2)
end

function BCLeft(obj::SolverCSD,n::Int)
    if obj.settings.problem == "WaterPhantomEdgar"
        E0 = obj.settings.eMax;
        E = obj.csd.eGrid[n];
        PsiLeft = 10^5*exp.(-200.0*(1.0.-obj.mu).^2)*exp(-50*(E0-E)^2)
        uHat = zeros(obj.settings.nPN)
        for i = 1:obj.settings.nPN
            uHat[i] = sum(PsiLeft.*obj.w.*obj.P[:,i]);
        end
        return uHat*obj.density[1]*obj.csd.S[n]
    else
        return 0.0
    end
end

function Slope(u,v,w,dx)
    mMinus = (v-u)/dx;
    mPlus = (w-v)/dx;
    if mPlus*mMinus > 0
        return 2*mMinus*mPlus/(mMinus+mPlus);
    else
        return 0.0;
    end
end

function RhsHighOrder(obj::SolverCSD,u::Array{Float64,3},t::Float64=0.0)   
    #Boundary conditions
    #obj.outRhs[1,:] = u[1,:];
    #obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];
    dx = obj.settings.dx
    dz = obj.settings.dy

    for j=2:obj.settings.NCellsX-1
        for i=2:obj.settings.NCellsY-1
            # Idea: use this formulation to define outRhsX and outRhsY, then apply splitting to get u2 = u + dt*outRhsX(u), uNew = u2 + dt*outRhsY(u2)
            obj.outRhs[j,i,:] = 1/2/dx * obj.pn.Ax * (u[j+1,i,:]/obj.density[j+1,i]-u[j-1,i,:]/obj.density[j-1,i]) - 1/2/dx * obj.AbsAx*( u[j+1,i,:]/obj.density[j+1,i] - 2*u[j,i,:]/obj.density[j,i] + u[j-1,i,:]/obj.density[j-1,i] );
            obj.outRhs[j,i,:] += 1/2/dz * obj.pn.Az * (u[j,i+1,:]/obj.density[j,i+1]-u[j,i-1,:]/obj.density[j,i-1]) - 1/2/dz * obj.AbsAz*( u[j,i+1,:]/obj.density[j,i+1] - 2*u[j,i,:]/obj.density[j,i] + u[j,i-1,:]/obj.density[j,i-1] );
        end
    end
    return obj.outRhs;
end

function RhsMatrix(obj::SolverCSD,u::Array{Float64,3},t::Float64=0.0)   
    #Boundary conditions
    #obj.outRhs[1,:] = u[1,:];
    #obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];
    dx = obj.settings.dx
    dz = obj.settings.dy

    for j=2:obj.settings.NCellsX-1
        for i=2:obj.settings.NCellsY-1
            # Idea: use this formulation to define outRhsX and outRhsY, then apply splitting to get u2 = u + dt*outRhsX(u), uNew = u2 + dt*outRhsY(u2)
            obj.outRhs[j,i,:] = 1/2/dx * obj.pn.Ax * (u[j+1,i,:]/obj.density[j+1,i]-u[j-1,i,:]/obj.density[j-1,i]) - 1/2/dx * obj.AbsAx*( u[j+1,i,:]/obj.density[j+1,i] - 2*u[j,i,:]/obj.density[j,i] + u[j-1,i,:]/obj.density[j-1,i] );
            obj.outRhs[j,i,:] += 1/2/dz * obj.pn.Az * (u[j,i+1,:]/obj.density[j,i+1]-u[j,i-1,:]/obj.density[j,i-1]) - 1/2/dz * obj.AbsAz*( u[j,i+1,:]/obj.density[j,i+1] - 2*u[j,i,:]/obj.density[j,i] + u[j,i-1,:]/obj.density[j,i-1] );
        end
    end
    return obj.outRhs;
end

function Rhs(obj::SolverCSD,u::Array{Float64,2},t::Float64=0.0)   

    return obj.L2x*u*obj.pn.Ax' + obj.L2y*u*obj.pn.Az' + obj.L1x*u*obj.AbsAx' + obj.L1y*u*obj.AbsAz';
end

function Solve(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);

    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    uNew = deepcopy(u)

    prog = Progress(nEnergies,1)

    out = zeros(nx*ny,N);
    vout = RhsMatrix(obj,v)
    for k = 1:N
        out[:,k] = vec(vout[:,:,k]');
    end

    println("error rhs = ",norm(Rhs(obj,u) - out))
    println("norm rhs = ",norm(Rhs(obj,u)))
    println("norm rhs = ",norm(out))

    #loop over energy
    for n=1:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # set boundary condition
        #u[1,:] .= BCLeft(obj,n);

        # perform time update
        uTilde = u .- dE * Rhs(obj,u); 

        # apply filtering
        #lam = 0.0#5e-7
        #for j = 1:(settings.NCellsX-1)
        #    for i = 1:(settings.NCellsY-1)
        #        for k = 1:obj.settings.nPN
        #            uTilde[j,i,k] = uTilde[j,i,k]/(1+lam*k^2*(k-1)^2);
        #        end
        #    end
        #end

        #uTilde[1,:] .= BCLeft(obj,n);
        #uNew = uTilde .- dE*uTilde*D;
        for j = 1:size(uNew,1)
            uNew[j,:] = (I + dE*D)\uTilde[j,:];
        end
        
        # update dose
        obj.dose .+= dE * uNew[:,1] * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));
        #if n > 1 && n < nEnergies
        #    obj.dose .+= 0.5 * dE * ( uNew[:,:,1] * S[n] + u[:,:,1] * S[n - 1] ) ./ obj.density;    # update dose with trapezoidal rule
            #obj.dose .+= dE * uNew[:,:,1] * SMinus[n] ./ obj.density;
        #else
        #    obj.dose .+= 0.5*dE * uNew[:,:,1] * S[n] ./ obj.density;
        #end

        u .= uNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose;

end

function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
end

function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*nx + j]
        end
    end
    return m;
end