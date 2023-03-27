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

    # problem definitions
    problem::String;

    #particle type
    particle::String;
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
    ϑ::Float64;  
    ϑIndex::Float64;
    cη::Float64;

    function Settings(Nx::Int=102,Ny::Int=102,r::Int=15,problem::String="LineSource",particle::String="Electrons")
        #Proton rest energy
        if particle == "Protons"
            eRest = 938.26 #MeV
        elseif particle == "Electrons"
            eRest = 0.5 #MeV -> estimate, look this up
        end
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
        ϑIndex = 1;
        ϑ = 1e-2;
        if problem =="LineSource"
            a = -1.5
            b = 1.5;
            c = -1.5;
            d = 1.5;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2);    
            eMax = 1.0
            ϑIndex = 0;
            ϑ = 0.3;#0.5;
            #ϑ = 1e-1;
        elseif problem =="2DHighLowD"
            a = 0.0
            b = 1.0;
            c = 0.0;
            d = 1.0;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*2.5;    
            eMax = 1.0
            ϑIndex = 0;
            ϑ = 0.3;#0.5;
            density[Int(floor(NCellsX*0.5/(b-a))):end,:] .= 5.0;
            density[Int(floor(NCellsX*0.55/(b-a))):end,:] .= 1.0;
            density[1:Int(floor(NCellsX*0.45/(b-a))),:] .= 7.0;
            density[Int(floor(NCellsX*0.55/(b-a))):end,Int(floor(NCellsY*0.52/(b-a))):end] .= 20.0;
        elseif problem =="2DHighD"
            a = 0.0
            b = 1.0;
            c = 0.0;
            d = 1.0;
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*0.5;    
            eKin = 80;  
            eMax = eKin + eRest
            ϑIndex = 0;
            ϑ = 0.3;#0.5;
            density[Int(floor(NCellsX*0.56/(b-a))):end,:] .= 5.0;
        elseif problem =="validation"
            pathlib = pyimport("pathlib")
            path = pathlib.Path(pwd())
            println(path)
            img = Float64.(Gray.(load("5-070_part.png")))
            nx = size(img,1)
            ny = size(img,2)
            densityMin = 0.2

            for i = 1:NCellsX
                for j = 1:NCellsY
                    idx1 = max( Int(floor(i/(NCellsX)*nx)), 1);
                    idx2 = max( Int(floor(j/(NCellsY)*ny)), 1);
                    density[i,j] = max(1.85*img[idx1,idx2],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                end
            end
            #density = ones(size(density))

            a = 0.0; # left boundary
            b = 14.5; # right boundary
            c = 0.0; # lower boundary
            d = 14.5; # upper boundary
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*120.5;  
            eKin = 85;
            eMax = eKin + eRest
            ϑIndex = 0;
            ϑ = 0.3;#0.5;

            Omega1 = -1.0;
            Omega3 = 0.1;
            x0 = 0.95*b;
            y0 = 0.5*d; 
            #Omega1 = -0.0;
            #Omega3 = 1.0;
            #x0 = 0.5*b;
            #y0 = 0.5*d;
        elseif problem =="waterBeamElectrons"
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 3.0; # upper boundary
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*2.0;  
            eKin = 21;
            eMax = eKin + eRest
            ϑIndex = 0;
            ϑ = 0.3;#0.5;

            Omega1 = -1.0;
            Omega3 = 0.0;
            x0 = 0.5*b;
            y0 = 0.05*d; 
            Omega1 = -0.0;
            Omega3 = 1.0;

            # generate grid
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
            idxX = findall( xMid .> 1.0 )
            idxY = findall( (yMid .> 5.0) .& (yMid .< 7.0) )
            density[idxX, idxY] .= 5.0
        elseif problem =="waterBeam"
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 8.0; # upper boundary
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*120.5;  
            eKin = 85;
            eMax = eKin + eRest
            ϑIndex = 0;
            ϑ = 0.3;#0.5;

            Omega1 = -1.0;
            Omega3 = 0.0;
            x0 = 0.5*b;
            y0 = 0.05*d; 
            Omega1 = -0.0;
            Omega3 = 1.0;

            # generate grid
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
            idxX = findall( xMid .> 1.0 )
            idxY = findall( (yMid .> 5.0) .& (yMid .< 7.0) )
            density[idxX, idxY] .= 5.0
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
            eMax = 21.0
            cfl = 1.5
            x0 = 0.5*b;
            y0 = 1.0*d;
            Omega1 = -1.0;
            Omega3 = -1.0;
            ϑ = 1e-3;
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
        dE = cfl*min(dx,dy)*minimum(density);#1/312;#cfl*min(dx,dy)*minimum(density);
        
        # number PN moments
        nPN = 21#7, 13, 21; # use odd number

        cη = 5;

        # build class
        new(Nx,Ny,NCellsX,NCellsY,a,b,c,d,dx,dy,eMax,eRest,dE,cfl,nPN,x,xMid,y,yMid,problem,particle,x0,y0,Omega1,Omega3,densityMin,sigmaT,sigmaS,density,r,ϑ,ϑIndex,cη);
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
