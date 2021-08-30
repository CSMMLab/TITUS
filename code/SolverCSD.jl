__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK

include("CSD.jl")

struct SolverCSD
    # spatial grid of cell interfaces
    x::Array{Float64};

    # Solver settings
    settings::Settings;

    # preallocate memory for performance
    outRhs::Array{Float64,2};
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};
    # Roe matrix
    AbsA::Array{Float64,2};
    # normalized Legendre Polynomials
    P::Array{Float64,2};
    # quadrature points
    mu::Array{Float64,1};
    w::Array{Float64,1};

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;

    # functionalities of the CSD approximation
    csd::CSD;

    # material density
    density::Array{Float64,1};

    # dose vector
    dose::Array{Float64,1};

    # constructor
    function SolverCSD(settings)
        x = settings.x;

        outRhs = zeros(settings.NCells,settings.nPN);

        # setup flux matrix
        gamma = zeros(settings.nPN);
        for i = 1:settings.nPN
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

        # setup Roe matrix
        S = eigvals(A)
        V = eigvecs(A)
        AbsA = V*abs.(diagm(S))*inv(V)

        println("check: ",maximum(A.-V*diagm(S)*inv(V)))

        # construct CSD fields
        csd = CSD(settings);

        # set density vector
        density = settings.density;

        # allocate dose vector
        dose = zeros(settings.NCells)

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

        new(x,settings,outRhs,gamma,A,AbsA,P,mu,w,settings.sigmaT,settings.sigmaS,csd,density,dose);
    end
end

function SetupIC(obj::SolverCSD)
    u = zeros(obj.settings.NCells,obj.settings.nPN);
    if obj.settings.problem == "WaterPhantomKerstin" || obj.settings.problem == "AirCavity"
        PCurrent = collectPl(1,lmax=obj.settings.nPN-1);
        for i = 1:obj.settings.nPN
            u[:,i] = IC(obj.settings,obj.settings.xMid) .* PCurrent[i-1]/sqrt(obj.gamma[i])
        end
        println(maximum(u[:,1]))
    elseif obj.settings.problem == "WaterPhantomEdgar"
        # Nx interfaces, means we have Nx - 1 spatial cells
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

function Rhs(obj::SolverCSD,u::Array{Float64,2},t::Float64=0.0)   
    #Boundary conditions
    obj.outRhs[1,:] = u[1,:];
    obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];
    dx = obj.settings.dx
    dt = obj.settings.dE

    for j=2:obj.settings.NCells-1
        #densityLeft = 2*obj.density[j-1]*obj.density[j]/(obj.density[j-1]+obj.density[j])
        #densityRight = 2*(obj.density[j+1]*obj.density[j])/(obj.density[j+1]+obj.density[j])
        #obj.outRhs[j,:] = -1/(2*dt)*(u[j+1,:]-2*u[j,:]+u[j-1,:])+0.5 * obj.A*((u[j+1,:]+u[j,:])/densityRight-(u[j-1,:]+u[j,:])/densityLeft)/dx;
        #obj.outRhs[j,:] = -1/(2*dt)*(u[j+1,:]-2*u[j,:]+u[j-1,:])+0.5 * obj.A*((u[j+1,:]/obj.density[j+1]+u[j,:]/obj.density[j])-(u[j-1,:]/obj.density[j-1]+u[j,:]/obj.density[j]))/dx;
        obj.outRhs[j,:] = 1/2/dx * (obj.A*(u[j+1,:]/obj.density[j+1]-u[j-1,:]/obj.density[j-1]) - obj.AbsA * ( u[j+1,:]/obj.density[j+1] - 2*u[j,:]/obj.density[j] + u[j-1,:]/obj.density[j-1] ))
        #obj.outRhs[j,:] = 1/2/dx * (obj.A*(u[j+1,:]-u[j-1,:]) - obj.AbsA * ( u[j+1,:] - 2*u[j,:] + u[j-1,:] ))
    end
    return obj.outRhs;
end

function Solve(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

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

    #loop over energy
    for n=1:(nEnergies-0)
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
        D = Diagonal(sigmaS[1] .- sigmaS);

        # set boundary condition
        u[1,:] .= BCLeft(obj,n);

        # perform time update
        uTilde = u .- dE * Rhs(obj,u); 

        # apply filtering
        lam = 0.0#5e-7
        for j = 1:(settings.NCells-1)
            for i = 1:obj.settings.nPN
                uTilde[j,i] = uTilde[j,i]/(1+lam*i^2*(i-1)^2);
            end
        end

        uTilde[1,:] .= BCLeft(obj,n);
        #uNew = uTilde .- dE*uTilde*D;
        for j = 1:(settings.NCells-1)
            uNew[j,:] = (I + dE*D)\uTilde[j,:];
        end
        
        # update dose
        if n > 1
            obj.dose .+= 0.5 * dE * ( uNew[:,1] * S[n] + u[:,1] * S[n - 1] ) ./ obj.density;    # update dose with trapezoidal rule
        else
            obj.dose .+= dE * uNew[:,1] * S[n] ./ obj.density;
        end

        u .= uNew;
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose;

end