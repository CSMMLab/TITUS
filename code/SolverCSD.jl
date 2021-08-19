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

        # construct CSD fields
        csd = CSD(settings);

        # set density vector
        density = ones(settings.NCells)

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

        new(x,settings,outRhs,gamma,A,P,mu,w,settings.sigmaT,settings.sigmaS,csd,density,dose);
    end
end

function SetupIC(obj::SolverCSD)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    return u;
end

function PsiLeft(obj::SolverCSD,n::Int,mu::Float64)
    E0 = 5.0;
    return 10^5*exp(-200.0*(1.0-mu)^2)*exp(-50*(E0-E)^2)
end


function BCLeft(obj::SolverCSD,n::Int)
    E0 = 5.0;
    E = obj.csd.eGrid[n];
    PsiLeft = 10^5*exp.(-200.0*(1.0.-obj.mu).^2)*exp(-50*(E0-E)^2)
    uHat = zeros(obj.settings.nPN)
    for i = 1:obj.settings.nPN
        uHat[i] = sum(PsiLeft.*obj.w.*obj.P[:,i]);
    end
    return uHat*obj.density[1]*obj.csd.S[n]
end

function Rhs(obj::SolverCSD,u::Array{Float64,2},t::Float64=0.0)   
    #Boundary conditions
    obj.outRhs[1,:] = u[1,:];
    obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];
    dx = obj.settings.dx

    for j=2:obj.settings.NCells-1
        obj.outRhs[j,:] = (0.5 * (obj.A*u[j,:]+obj.A*u[j+1,:]) - obj.settings.dx/(2*obj.settings.dt)*(u[j+1,:]-u[j,:]) - 0.5 * (obj.A*u[j-1,:]+obj.A*u[j,:]) + obj.settings.dx/(2*obj.settings.dt)*(u[j,:]-u[j-1,:]))/obj.settings.dx;
        # simplifying Pias code gives, so it should be correct: obj.outRhs[j,:] = -1/(2*dt)*(u[j+1,:]-2*u[j,:]+u[j-1,:])+0.5 * (obj.A*u[j+1,:]-obj.A*u[j-1,:])/dx;
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

    #Precompute quadrature points, weights and polynomials at quad. points
    Nq=200;
    (x,w) = gauss(Nq);
    P=zeros(Nq,obj.settings.nPN);
    for k=1:Nq
        P[k,:] = collectPl(x[k],lmax=obj.settings.nPN-1);
    end

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings
    gamma = obj.gamma
    

    #Compute flux matrix A 
    for i=1:obj.settings.nPN
        for l=1:obj.settings.nPN
            for k=1:Nq
              obj.A[i,l] = obj.A[i,l] + w[k]*x[k]*P[k,i]*P[k,l]/sqrt(gamma[l])/sqrt(gamma[i]);
            end 
        end
    end

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dt = dE

    # setup flux matrix (alternative analytic computation)
    #A = zeros(settings.nPN,settings.nPN)

    #for i = 1:(settings.nPN-1)
    #    n = i-1;
    #    A[i,i+1] = (n+1)/(2*n+1)*sqrt(gamma[i+1])/sqrt(gamma[i]);
    #end

    #for i = 2:settings.nPN
    #    n = i-1;
    #    A[i,i-1] = n/(2*n+1)*sqrt(gamma[i-1])/sqrt(gamma[i]);
    #end

    uNew = deepcopy(u)

    #loop over energy
    for n=1:(nEnergies-0)
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
        D = Diagonal(sigmaS[1] .- sigmaS);

        # set boundary condition
        u[1,:] .= BCLeft(obj,n);

        # perform time update
        uTilde = u .- dE * Rhs(obj,densityInv*u); 
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