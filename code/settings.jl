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
    # number of collocation points
    Nxi::Int;
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

    # uncertainty
    rho0Inv::Array{Float64,2};
    rho1Inv::Array{Float64,2};
    rho0InvVec::Array{Float64,1};
    rho1InvVec::Array{Float64,1};

    rhoInv::Array{Float64,1};
    rhoInvX::Array{Float64,2};
    rhoInvXi::Array{Float64,2};

    # parameters for rank adaptivity
    rMin::Int;
    rMax::Array{Int64,1};
    ε::Float64;

    function Settings(Nx::Int=102,Ny::Int=102,Nxi::Int=100,r::Int=15,problem::String="LineSource")

        # spatial grid setting
        NCellsX = Nx - 1;
        NCellsY = Ny - 1;

        a = 0.0; # left boundary
        b = 14.5; # right boundary

        c = 0.0; # lower boundary
        d = 14.5; # upper boundary

        density = ones(NCellsX,NCellsY);
        rhoInvX = zeros(2,2); rhoInv = zeros(2); rhoInvXi = zeros(2,2);

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

            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2);    
            eMax = 1.0
            nx = NCellsX;
            ny = NCellsY;
            xi = collect(range(0,1,Nxi));
            
            rXi = 2;
            σ = 1.0;
            ei = zeros(size(xi)); ei[3] = 1;
            fx = 1 .* vec(1 .+ sin.(4 .* xMid * yMid'));#ones(nx*ny);#1 .* vec(1 .+ sin.(4 .* xMid * yMid'))
            rhoInvX,rhoInv,rhoInvXi = svd((ones(nx*ny,Nxi).+ σ * xi' .* ones(nx*ny,Nxi) .* fx).^(-1))
            #rhoInvX,rhoInv,rhoInvXi = svd((ones(nx*ny,Nxi).+ σ * ei' .* ones(nx*ny,Nxi) .* fx).^(-1))
            rhoInvXi = Matrix(rhoInvXi);
            rhoInv = rhoInv[1:rXi];
            rhoInvX = rhoInvX[:,1:rXi];
            rhoInvXi = rhoInvXi[:,1:rXi]
            #epsAdapt = 1e-1;
        elseif problem =="2DHighLowD"
            a = 0.0
            b = 1.0;
            c = 0.0;
            d = 1.0;
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
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*2.5;    
            eMax = 1.0
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            density[Int(floor(NCellsX*0.56/(b-a))):end,:] .= 5.0;
        elseif problem =="validation"
            a = 0.0483333333333333; # left boundary
            b = 14.4516666666667; # right boundary
            c = 0.0483333333333333; # lower boundary
            d = 14.4516666666667; # upper boundary
            sigmaS = 1.0;
            sigmaA = 0.0;  
            cfl = 0.99/sqrt(2)*2.5;    
            eMax = 40.0
            adaptIndex = 0;
            epsAdapt = 0.3;#0.5;
            Omega1 = 0.0;
            Omega3 = 1.0;
            x0 = 0.5*b;
            y0 = 0.0*d;
            #epsAdapt = 1e-1;
            density[Int(floor(NCellsX*0.5))+1:end,:] .= 1.85;
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
            Omega1 = 0.0;
            Omega3 = -1.0;
            epsAdapt = 1e-3;
            normOmega = sqrt(Omega1^2 + Omega3^2); Omega1 /= normOmega; Omega3 /= normOmega;
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
            eMax = 21.0
            cfl = 1.5
            x0 = 0.5*b;
            y0 = 1.0*d;
            Omega1 = -1.0;
            Omega3 = -1.0;
        elseif problem =="lung2"
            #img = Float64.(Gray.(load("phantom.png")))
            pathlib = pyimport("pathlib")
            path = pathlib.Path(pwd())
            println(path)
            img = Float64.(Gray.(load("Lung_square.png")))
            nx = size(img,1)
            ny = size(img,2)
            println(size(img))
            densityMin = 0.05
            for i = 1:NCellsX
                for j = 1:NCellsY
                    density[i,j] = max(1.85*img[Int(floor(i/NCellsX*nx)),Int(floor(j/NCellsY*ny))],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                end
            end
            b = 6.0; # right boundary
            d = 6.0; # upper boundary
            eMax = 20.0
            cfl = 1.5
            x0 = 0.0;
            y0 = 2.5;
            Omega1 = -1.0;
            Omega3 = 0.0;
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
        elseif problem =="timeCT"
            nx = NCellsX;
            ny = NCellsY;
            densityTmp = zeros(nx,ny);
            ndata = 10;
            densityInv = zeros(nx*ny,ndata);
            for k = 1:ndata
                img = Float64.(Gray.(load("CTData/$(k)-070.png")));
                #img = Float64.(Gray.(load("CTData/1-070.png")));
                nxi = size(img,1);
                nyi = size(img,2);
                densityMin = 0.2;
                #img = ones(size(img))
                for i = 1:nx
                    for j = 1:ny
                        densityTmp[i,j] = max(1.85*img[max(Int(floor(i/nx*nxi)),1),max(Int(floor(j/ny*nyi)),1)],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
                    end
                end
                densityInv[:,k] = 1.0./Mat2Vec(densityTmp);
            end
            
            xi_tab = collect(range(0,1,10));
            xi = collect(range(0,1,Nxi));
            densityInvF = zeros(nx*ny,Nxi);
            
            for j = 1:nx*ny
                xiToDensity = LinearInterpolation(xi_tab, densityInv[j,:]; extrapolation_bc=Throw())
                for i = 1:Nxi  
                    densityInvF[j,i] = xiToDensity(xi[i])
                end
            end

            #densityInvF = ones(size(densityInvF)) # CHANGED: try out constant density
            
            rXi = 5;
            rhoInvX,rhoInv,rhoInvXi = svd(densityInvF)
            rhoInvXi = Matrix(rhoInvXi);
            rhoInv = rhoInv[1:rXi];
            rhoInvX = rhoInvX[:,1:rXi];
            rhoInvXi = rhoInvXi[:,1:rXi]


            println("error tissue is ",norm(rhoInvX*Diagonal(rhoInv)*rhoInvXi' - densityInvF))

            b = 14.5; # right boundary
            d = 14.5; # upper boundary
            eMax = 21.0
            cfl = 0.6
            x0 = 0.8*b;
            y0 = 1.0*d;
            Omega1 = 0.8;
            #Omega1 = 0.0;
            Omega3 = -1.0;
            normOmega = sqrt(Omega1^2 + Omega3^2); Omega1 /= normOmega; Omega3 /= normOmega;
        elseif problem =="deterministic"
            nx = NCellsX;
            ny = NCellsY;
            
            rXi = 1;
            rhoInvX,rhoInv,rhoInvXi = svd(ones(nx*ny,Nxi))
            rhoInvXi = Matrix(rhoInvXi);
            rhoInv = rhoInv[1:rXi];
            rhoInvX = rhoInvX[:,1:rXi];
            rhoInvXi = rhoInvXi[:,1:rXi]

            b = 14.5; # right boundary
            d = 14.5; # upper boundary
            eMax = 21.0
            cfl = 0.6
            x0 = 0.8*b;
            y0 = 1.0*d;
            Omega1 = 0.8;
            Omega3 = -1.0;
            normOmega = sqrt(Omega1^2 + Omega3^2); Omega1 /= normOmega; Omega3 /= normOmega;
        end
        sigmaT = sigmaA + sigmaS;

        # define inverse tissue density
        rho0Inv = 2*ones(NCellsX,NCellsY); #1.0./density;#
        rho1Inv = 1.0*ones(NCellsX,NCellsY);
        #density = ones(size(density));
        #densityMin = 1.0

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
        #dE = cfl*min(dx,dy)/maximum(rho0Inv .+ rho1Inv);#1/90#
        dE = cfl*min(dx,dy)*densityMin;#1/90#
        
        # number PN moments
        nPN = 11#13, 21; # use odd number

        rMin = 2;
        rMax = [Int(floor(NCellsX*NCellsY/2)); Int(floor((nPN+1)^2/2)); Int(floor(Nxi/2))];
        ε = 1e-5;

        # build class
        new(Nx,Ny,NCellsX,NCellsY,Nxi,a,b,c,d,dx,dy,eMax,dE,cfl,nPN,x,xMid,y,yMid,problem,x0,y0,Omega1,Omega3,densityMin,sigmaT,sigmaS,density,r,epsAdapt,adaptIndex,rho0Inv,rho1Inv,vec(rho0Inv),vec(rho1Inv),rhoInv,rhoInvX,rhoInvXi,rMin,rMax,ε);
    end
end

function IC(obj::Settings,x,y)
    out = zeros(length(x),length(y));
    posBeamX = (obj.b+obj.a)/2;
    posBeamY = (obj.d+obj.c)/2;
    if obj.problem != "LineSource" && obj.problem != "2DHighD" && obj.problem != "2DHighLowD"
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
