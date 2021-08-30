__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # number spatial cells
    NCells::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    dx::Float64

    # time settings
    # end time
    eMax::Float64;
    # time increment
    dE::Float64;
    # CFL number 
    cfl::Float64;
    
    # degree PN
    nPN::Int64;

    # spatial grid
    x
    xMid

    # problem definitions
    problem::String;

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;    

    # patient density
    density::Array{Float64,1};

    function Settings(Nx::Int=502,problem::String="LineSource")

        # spatial grid setting
        NCells = Nx - 1;

        a = -2.5; # left boundary
        b = 3.0; # right boundary

        problem = "WaterPhantomKerstin" # WaterPhantomKerstin, AirCavity

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0; 
        if problem =="LineSource"
            sigmaS = 1.0;
            sigmaA = 0.0;        
        end
        sigmaT = sigmaA + sigmaS;

        density = ones(NCells);
        if problem =="BoneLung"
            # determin regions for different organs
            x0 = 1.5; x1 = 2.0;
            i0 = Integer(floor(NCells*x0/b)); 
            i1 = Integer(floor(NCells*x1/b))
            density[1:i0] .= 1.04;
            density[(i0+1):i1] .= 1.85;
            density[(i1+1):end] .= 0.3;
        elseif problem == "AirCavity"
            # determin regions for air cavity
            x0 = 1.5; x1 = 2.0;
            b = 3.5
            i0 = Integer(floor(NCells*(x0-a)/(b-a))); 
            i1 = Integer(floor(NCells*(x1-a)/(b-a)))
            density[(i0+1):i1] .= 0.01;
        end

        # spatial grid
        x = collect(range(a,stop = b,length = NCells));
        dx = x[2]-x[1];
        x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        xMid = x[1:(end-1)].+0.5*dx

        # time settings
        eMax = 5.0
        cfl = 1.9#1.0#1.2#0.6#1.9; # CFL condition
        dE = cfl*dx*minimum(density);
        
        # number PN moments
        nPN = 4; 

        # build class
        new(Nx,NCells,a,b,dx,eMax,dE,cfl,nPN,x,xMid,problem,sigmaT,sigmaS,density);
    end

end

function IC(obj::Settings,x)
    y = zeros(size(x));
    
    x0 = 0.0
    s1 = 0.1
    s2 = s1^2
    floor = 1e-4
    x0 = 0.0
    for j = 1:length(y);
        y[j] = 1/(s1*sqrt(2*pi))*exp.(-x[j].^2 ./ 2.0./s2^2)
    end
    
    return y;
end