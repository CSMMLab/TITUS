__precompile__

using ProgressMeter
using LinearAlgebra

struct Solver
    # spatial grid of cell interfaces
    x::Array{Float64,1};

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
        gamma = zeros(settings.nPN)
        A = zeros(settings.nPN,settings.nPN)


        new(x,settings,outRhs,gamma,A,settings.sigmaT,settings.sigmaS);
    end
end

function SetupIC(obj::Solver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = 2.0/sqrt(obj.gamma[1])*IC(obj.settings,obj.settings.xMid);
    return u;
end

function Rhs(obj::Solver,u::Array{Float64,2},t::Float64=0.0)   


    return obj.outRhs;
end

function Solve(obj::Solver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u = SetupIC(obj);

    # return end time and solution
    return t, 0.5*sqrt(obj.gamma[1])*u;

end