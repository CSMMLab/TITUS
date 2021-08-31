__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK

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

    # dose vector
    dose::Array{Float64,2};

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
        dose = zeros(settings.NCellsX,settings.NCellsY)

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

        new(x,y,settings,outRhs,gamma,AbsAx,AbsAz,P,mu,w,csd,pn,density,dose);
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
    elseif obj.settings.problem == "2D"
        u[:,:,1] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid) 
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

function Rhs(obj::SolverCSD,u::Array{Float64,3},t::Float64=0.0)   
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

function Solve(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S; # todo

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);

    # Set up initial condition
    u = SetupIC(obj);

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    uNew = deepcopy(u)

    prog = Progress(nEnergies,1)

    #loop over energy
    for n=1:(nEnergies-0)
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
        for j = 1:(settings.NCellsX-1)
            for i = 1:(settings.NCellsY-1)
                uNew[j,i,:] = (I + dE*D)\uTilde[j,i,:];
                #uNew[j,i,:] = uTilde[j,i,:]
            end
        end
        
        # update dose
        if n > 1
            obj.dose .+= 0.5 * dE * ( uNew[:,:,1] * S[n] + u[:,:,1] * S[n - 1] ) ./ obj.density;    # update dose with trapezoidal rule
        else
            obj.dose .+= dE * uNew[:,:,1] * S[n] ./ obj.density;
        end

        u .= uNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose;

end