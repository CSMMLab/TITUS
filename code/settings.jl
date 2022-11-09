__precompile__

using Images, FileIO

mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    Ny::Int64;
    Nz::Int64;
    # number spatial cells
    NCellsX::Int64;
    NCellsY::Int64;
    NCellsZ::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    c::Float64;
    d::Float64;
    e::Float64;
    f::Float64;
    # grid cell width
    dx::Float64
    dy::Float64
    dz::Float64

    # time settings
    # end time
    eMax::Float64;
    eRest::Float64;
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
    z
    zMid

    # problem definitions
    problem::String;

    #particle type
    particle::String;
    # beam properties
    x0::Float64;
    y0::Float64;
    z0::Float64;
    Omega1::Float64;
    Omega2::Float64;
    Omega3::Float64;
    densityMin::Float64;
    sigmaX::Float64; # spatial std of initial beam
    sigmaY::Float64; # spatial std of initial beam
    sigmaZ::Float64; # spatial std of initial beam
    sigmaE::Float64; # energy std of boundary beam

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;    

    # patient density
    density::Array{Float64,3};

    # rank
    r::Int;

    # tolerance for rank adaptivity
    epsAdapt::Float64;  
    adaptIndex::Float64;

    function Settings(Nx::Int=52,Ny::Int=52,Nz::Int=52,r::Int=15,problem::String="validation",particle::String="Protons")
        #Proton rest energy
        if particle == "Protons"
            eRest = 938.26 #MeV
        elseif particle == "Electrons"
            eRest = 0.5 #MeV -> estimate, look this up
        end
        # spatial grid setting
        NCellsX = Nx - 1;
        NCellsY = Ny - 1;
        NCellsZ = Nz - 1;

        a = 0.0; # left boundary
        b = 14.5; # right boundary

        c = 0.0; # lower boundary
        d = 14.5; # upper boundary

        e = 0.0; # left z boundary
        f = 14.5; # right z boundary

        density = ones(NCellsX,NCellsY,NCellsZ); 

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0;
        eMax = 1.0;
        x0 = 0.5*b;
        y0 = 1.0*d;
        z0 = 0.5*f;
        Omega1 = -1.0;
        Omega2 = -0.0;
        Omega3 = -1.0;
        densityMin = 0.2;
        adaptIndex = 1;
        epsAdapt = 1e-2;
        sigmaX = 0.1;
        sigmaY = 0.02;
        sigmaZ = 0.1;
        if problem =="LineSource"
            a = -1.5
            b = 1.5;
            c = -1.5;
            d = 1.5;
            e = -1.5;
            f = 1.5;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2);    
            eMax = 1.0
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            #epsAdapt = 1e-1;
        elseif problem =="2DHighLowD"
            a = 0.0
            b = 1.0;
            c = 0.0;
            d = 1.0;
            e = 0.0;
            f = 1.0;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*2.5;    
            eMax = 1.0
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            density[Int(floor(NCellsX*0.5/(b-a))):end,:] .= 5.0;
            density[Int(floor(NCellsX*0.55/(b-a))):end,:] .= 1.0;
            density[1:Int(floor(NCellsX*0.45/(b-a))),:] .= 7.0;
            density[Int(floor(NCellsX*0.55/(b-a))):end,Int(floor(NCellsY*0.52/(b-a))):end] .= 20.0;
        elseif problem =="2DHighD"
            a = 0.0
            b = 1.0;
            c = 0.0;
            d = 1.0;
            e = 0.0;
            f = 1.0;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*0.5;    
            eKin = 80;  
            eMax = eKin + eRest
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            density[Int(floor(NCellsX*0.56/(b-a))):end,:] .= 5.0;
        elseif problem =="validation"
            a = 0; # left boundary
            b = 2; # right boundary
            c = 0; # lower boundary
            d = 7.5; # upper boundary
            e = 0;
            f = 2;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*20.5* 1.3;  
            eKin = 90;
            eMax = eKin + eRest
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            Omega1 = 0.0;
            Omega2 = 1.0;
            Omega3 = 0.0;
            x0 = 0.5 * b;
            y0 = 0.5;
            z0 = 0.5 * f;
            sigmaE = 1e-2;
            #epsAdapt = 1e-1;
            #density[Int(floor(NCellsX*0.5)):end,:] .= 5.0; #beam hits interface
            #density[:,Int(floor(NCellsX*0.56)):end] .= 5.0; #beam perpendicular to interface
            #density[:,Int(floor(NCellsY*0.75)):Int(floor(NCellsY*0.8))] .= 5.0; #inserted box of high density 
        elseif problem =="lungOrig"
            #img = Float64.(Gray.(load("phantom.png")))
            pathlib = pyimport("pathlib")
            path = pathlib.Path(pwd())
            println(path)
            img = Float64.(Gray.(load("LungOrig.png")))
            nx = size(img,1)
            ny = size(img,2)
            densityMin = 0.2
            for i = 1:NCellsX
                for j = 1:NCellsY
                    density[i,j] = max(1.85*img[Int(floor(i/NCellsX*nx)),Int(floor(j/NCellsY*ny))],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                end
            end
            b = 14.5; # right boundary
            d = 18.5; # upper boundary
            f = 1.0;
            eMax = 21.0
            cfl = 1.5
            x0 = 0.5*b;
            y0 = 1.0*d;
            Omega1 = -1.0;
            Omega3 = -1.0;
            epsAdapt = 1e-3;
        elseif problem =="lung"
            #img = Float64.(Gray.(load("phantom.png")))
            pathlib = pyimport("pathlib")
            path = pathlib.Path(pwd())
            println(path)
            img = Float64.(Gray.(load("Lung.png")))
            nx = size(img,1)
            ny = size(img,2)
            println(size(img))
            densityMin = 0.05
            for i = 1:NCellsX
                for j = 1:NCellsY
                    density[i,j] = max(1.85*img[Int(floor(i/NCellsX*nx)),Int(floor(j/NCellsY*ny))],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                end
            end
            b = 14.5; # right boundary
            d = 14.5; # upper boundary
            f = 1.0;
            eKin = 21.0
            eMax = sqrt(eKin^2 + eRest^2)
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
            f = 1.0;
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
        z = collect(range(e,stop = f,length = NCellsZ));
        dz = z[2]-z[1];
        z = [z[1]-dz;z]; # add ghost cells so that boundary cell centers lie on a and b
        z = z.+dz/2;
        zMid = z[1:(end-1)].+0.5*dz

        # time settings
        #cfl = 1.5#1.4 # CFL condition
        dE = cfl*min(dx,dy,dz)*minimum(density);#1/312;#cfl*min(dx,dy)*minimum(density);
        
        # number PN moments
        nPN = 21#37, 21; # use odd number

        # build class
        new(Nx,Ny,Nz,NCellsX,NCellsY,NCellsZ,a,b,c,d,e,f,dx,dy,dz,eMax,eRest,dE,cfl,nPN,x,xMid,y,yMid,z,zMid,problem,particle,x0,y0,z0,Omega1,Omega2,Omega3,densityMin,sigmaX,sigmaY,sigmaZ,sigmaE,sigmaT,sigmaS,density,r,epsAdapt,adaptIndex);
    end
end

function IC(obj::Settings,x,y)
    out = zeros(length(x),length(y));
    posBeamX = (obj.b+obj.a)/2;
    posBeamY = (obj.d+obj.c)/2;
    if obj.problem != "LineSource"  && obj.problem != "2DHighLowD" && obj.problem != "2DHighD"
        return out;
    end
    x0 = x .- posBeamX;
    y0 = y .- posBeamY;
    
    s1 = 0.05
    if obj.problem == "2DHighD" || obj.problem == "2DHighLowD"
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

function IC(obj::Settings,x,y,z)
    out = zeros(length(x),length(y),length(z));
    posBeamX = (obj.b+obj.a)/2;
    posBeamY = (obj.d+obj.c)/2;
    posBeamZ = (obj.f+obj.e)/2;
    if obj.problem != "LineSource" && obj.problem != "2DHighD"
        return out;
    end
    x0 = x .- posBeamX;
    y0 = y .- posBeamY;
    z0 = z .- posBeamZ;
    
    s1 = 0.05
    if obj.problem == "2DHighD"
        s1 = 0.01
    end

    s2 = s1^2
    floor = 1e-4
    for j = 1:length(x)
        for i = 1:length(y)
            for k = 1:length(z)
                out[j,i,k] = 1/(s1*sqrt(2*pi))^2*exp.(-(x0[j].^2+y0[i].^2+z0[k].^2) ./ 2.0./s2)
            end
        end
    end
    
    return out;
end
