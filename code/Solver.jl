__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK

struct Solver
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

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;

    # constructor
    function Solver(settings)
        x = settings.x;

        outRhs = zeros(settings.NCells,settings.nPN);

        # setup flux matrix
        gamma = ones(settings.nPN);
        A = zeros(settings.nPN,settings.nPN);



        new(x,settings,outRhs,gamma,A,settings.sigmaT,settings.sigmaS);
    end
end

function SetupIC(obj::Solver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.gamma[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

function Rhs(obj::Solver,u::Array{Float64,2},t::Float64=0.0)   
    #Boundary conditions
    obj.outRhs[1,:] = u[1,:];
    obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];

    for j=2:obj.settings.NCells-1
        obj.outRhs[j,:] = (0.5 * (obj.A*u[j,:]+obj.A*u[j+1,:]) - obj.settings.dx/(2*obj.settings.dt)*(u[j+1,:]-u[j,:]) - 0.5 * (obj.A*u[j-1,:]+obj.A*u[j,:]) + obj.settings.dx/(2*obj.settings.dt)*(u[j,:]-u[j-1,:]))/obj.settings.dx;
    end
    return obj.outRhs;
end

function ScatteringMatrix(obj::Solver)
    G=-1*ones(obj.settings.nPN,1);
    nSteps= 100000;
    intSum = 0;
    for l=2:obj.settings.nPN
        G[l]=0;
        # for k=-l:l
        #     globalIdx = l*l+k+l+1;
        #     G[globalIdx]=0;
        #     int_Sum = 0.5 * (Pl(-1,l)+Pl(1,l));
        #     for i=1:nSteps
        #         intSum = intSum + Pl(-1+i*2/nSteps,l);
        #     end
        #     intSum= intSum*2/nSteps;
        #     intSum= intSum*0.5;
        #     G[globalIdx]= 1- intSum
        # end
    end
    return G
end

function Solve(obj::Solver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u = SetupIC(obj);

    #Compute diagonal of scattering matrix G
    G = ScatteringMatrix(obj);
    sigmaS=ones(obj.settings.NCells,1).*obj.settings.sigmaS;

    #Precompute quadrature points, weights and polynomials at quad. points
    Nq=20;
    (x,w) = gauss(Nq);
    P=zeros(Nq,obj.settings.nPN);
    for k=1:Nq
        P[k,:] = collectPl(x[k],lmax=obj.settings.nPN-1);
    end

    #Compute flux matrix A 
    for i=1:obj.settings.nPN
        for l=1:obj.settings.nPN
            for k=1:Nq
              obj.A[i,l] = obj.A[i,l] + w[k]*x[k]*P[k,i]*P[k,l];
            end 
        end
    end

    #loop over time
    for t=0:dt:tEnd
        u = u - dt * Rhs(obj,u,t) - dt*u.*(obj.sigmaT.+sigmaS*G'); 
    end
    # return end time and solution
    return t, u; #0.5*sqrt(obj.gamma[1])*u;

end