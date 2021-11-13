__precompile__

using Images, FileIO

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
    # beam properties
    x0::Float64;
    y0::Float64;
    Omega1::Float64;
    Omega3::Float64;
    densityMin::Float64;

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;    

    # patient density
    density::Array{Float64,2};

    # rank
    r::Int;

    # tolerance for rank adaptivity
    epsAdapt::Float64;  
    adaptIndex::Float64;

    function Settings(Nx::Int=102,Ny::Int=102,r::Int=15,problem::String="LineSource")

        # spatial grid setting
        NCellsX = Nx - 1;
        NCellsY = Ny - 1;

        a = 0.0; # left boundary
        b = 14.5; # right boundary

        c = 0.0; # lower boundary
        d = 14.5; # upper boundary

        density = ones(NCellsX,NCellsY);

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0;
        eMax = 1.0;
        x0 = 0.5*b;
        y0 = 1.0*d;
        Omega1 = -1.0;
        Omega3 = -1.0;
        densityMin = 0.2;
        adaptIndex = 1;
        epsAdapt = 1e-2;
        if problem =="LineSource"
            a = -1.5
            b = 1.5;
            c = -1.5;
            d = 1.5;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2);    
            eMax = 1.0
            #adaptIndex = 0;
            #epsAdapt = 0.3;#0.5;
            epsAdapt = 1e-2;
        elseif problem =="2DHighD"
            a = 0.0
            b = 1.0;
            c = 0.0;
            d = 1.0;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2);    
            eMax = 1.0
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            density[Int(floor(NCellsX*0.56/(b-a))):end,:] .= 5.0;
        elseif problem =="lung"
            #img = Float64.(Gray.(load("phantom.png")))
            pathlib = pyimport("pathlib")
            path = pathlib.Path(pwd())
            println(path)
            img = Float64.(Gray.(load("Lung.png")))
            nx = size(img,1)
            ny = size(img,2)
            densityMin = 0.05
            for i = 1:NCellsX
                for j = 1:NCellsY
                    density[i,j] = max(1.85*img[Int(floor(i/NCellsX*nx)),Int(floor(j/NCellsY*ny))],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                end
            end
            b = 14.5; # right boundary
            d = 14.5; # upper boundary
            eMax = 21.0
            cfl = 1.5
            x0 = 0.5*b;
            y0 = 1.0*d;
            Omega1 = -1.0;
            Omega3 = -1.0;
        elseif problem =="liver"
            #img = Float64.(Gray.(load("phantom.png")))
            pathlib = pyimport("pathlib")
            path = pathlib.Path(pwd())
            println(path)
            img = Float64.(Gray.(load("liver_cut.jpg")))
            nx = size(img,1)
            ny = size(img,2)
            densityMin = 0.05
            for i = 1:NCellsX
                for j = 1:NCellsY
                    density[i,j] = max(1.85*img[Int(floor(i/NCellsX*nx)),Int(floor(j/NCellsY*ny))],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                end
            end
            b = 35.0; # right boundary
            d = 35.0; # upper boundary
            eMax = 60.0
            cfl = 1.5
            x0 = 1.0*b;
            y0 = 0.35*d;
            Omega1 = -1.0;
            Omega3 = -1.0;
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
        #cfl = 1.5#1.4 # CFL condition
        dE = cfl*dx*minimum(density);
        
        # number PN moments
        nPN = 21#13, 21; # use odd number

        # build class
        new(Nx,Ny,NCellsX,NCellsY,a,b,c,d,dx,dy,eMax,dE,cfl,nPN,x,xMid,y,yMid,problem,x0,y0,Omega1,Omega3,densityMin,sigmaT,sigmaS,density,r,epsAdapt,adaptIndex);
    end
end

function IC(obj::Settings,x,y)
    out = zeros(length(x),length(y));
    posBeamX = (obj.b+obj.a)/2;
    posBeamY = (obj.d+obj.c)/2;
    if obj.problem != "LineSource" && obj.problem != "2DHighD"
        return out;
    end
    x0 = x .- posBeamX;
    y0 = y .- posBeamY;
    
    s1 = 0.05
    if obj.problem == "2DHighD"
        s1 = 0.01
    end

    s2 = s1^2
    floor = 1e-4
    for j = 1:length(x);
        for i = 1:length(y);
            out[j,i] = 1/(s1*sqrt(2*pi))^2*exp.(-(x0[j].^2+y0[i].^2) ./ 2.0./s2)
        end
    end
    
    return out;
end
