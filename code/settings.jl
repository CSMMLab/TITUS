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
    tEnd::Float64;
    # time increment
    dt::Float64;
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

    function Settings(Nx::Int=502,problem::String="LineSource")
        # spatial grid setting
        NCells = Nx - 1;
        a = 0.0; # left boundary
        b = 3.0; # right boundary
        x = collect(range(a,stop = b,length = NCells));
        dx = x[2]-x[1];
        x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        xMid = x[1:(end-1)].+0.5*dx
        
        # time settings
        tEnd = 1.0#1.0;# 0.7;
        cfl = 10.1; # CFL condition
        dt = cfl*dx;
        
        # number PN moments
        nPN = 10; 

        # physical parameters
        if problem =="LineSource"
            sigmaS = 1.0;
            sigmaA = 0.0;        
        end
        sigmaT = sigmaA + sigmaS;

        # build class
        new(Nx,NCells,a,b,dx,tEnd,dt,cfl,nPN,x,xMid,problem,sigmaT,sigmaS);
    end

end

function IC(obj::Settings,x,xi=0.0)
    y = zeros(size(x));
    
    x0 = 0.0
    s1 = 0.03
    s2 = s1^2
    floor = 1e-4
    x0 = 0.0
    for j = 1:length(y);
        y[j] = max(floor,1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2))
    end
    
    return y;
end