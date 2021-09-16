__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    Ny::Int64;
    # number spatial cells
    NCellsX::Int64;
    NCellsY::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    c::Float64;
    d::Float64;
    # grid cell width
    dx::Float64
    dy::Float64

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
    y
    yMid

    # problem definitions
    problem::String;

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;    

    # patient density
    density::Array{Float64,2};

    function Settings(Nx::Int=102,Ny::Int=102,problem::String="LineSource")

        # spatial grid setting
        NCellsX = Nx - 1;
        NCellsY = Ny - 1;

        a = 0.0; # left boundary
        b = 1.0; # right boundary

        c = 0.0; # lower boundary
        d = 1.0; # upper boundary

        problem = "2DHighD" # WaterPhantomKerstin, AirCavity, 2D, 2DHighD

        density = ones(NCellsX,NCellsY);

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0; 
        if problem =="LineSource"
            sigmaS = 1.0;
            sigmaA = 0.0;        
        elseif problem =="2DHighD"
            density[Int(floor(NCellsX*0.56/(b-a))):end,:] .= 5.0;
        end
        sigmaT = sigmaA + sigmaS;

        # spatial grid
        x = collect(range(a,stop = b,length = NCellsX));
        dx = x[2]-x[1];
        x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        xMid = x[1:(end-1)].+0.5*dx
        y = collect(range(c,stop = d,length = NCellsY));
        dy = y[2]-y[1];
        y = [y[1]-dy;y]; # add ghost cells so that boundary cell centers lie on a and b
        y = y.+dy/2;
        yMid = y[1:(end-1)].+0.5*dy

        # time settings
        eMax = 1.0
        cfl = 1.7#1.4#1.2#0.6#1.9; # CFL condition
        dE = cfl*dx*minimum(density);
        
        # number PN moments
        nPN = 13; # use odd number

        # build class
        new(Nx,Ny,NCellsX,NCellsY,a,b,c,d,dx,dy,eMax,dE,cfl,nPN,x,xMid,y,yMid,problem,sigmaT,sigmaS,density);
    end
end

function IC(obj::Settings,x,y)
    posBeamX = 0.5;
    posBeamY = 0.5;
    x0 = x .- posBeamX;
    y0 = y .- posBeamY;
    out = zeros(length(x),length(y));
    s1 = 0.01
    s2 = s1^2
    floor = 1e-4
    for j = 1:length(x);
        for i = 1:length(y);
            out[j,i] = 1/(s1*sqrt(2*pi))^2*exp.(-(x0[j].^2+y0[i].^2) ./ 2.0./s2)
        end
    end
    
    return out;
end