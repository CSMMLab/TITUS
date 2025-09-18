__precompile__
include("utils.jl")
using Images, FileIO, TOML, Interpolations, MAT, DICOM

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
    eMax
    eMin
    eRest
    # time increment
    dE::Float64;
    # CFL number 
    cfl::Float64;
    #number beam energies
    N_E
    
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
    x0 #::Array{Float64,1};
    y0 #::Array{Float64,1};
    z0 #::Array{Float64,1};
    Omega1 #::Array{Float64,1};
    Omega2 #::Array{Float64,1};
    Omega3 #::Array{Float64,1};
    OmegaMin::Float64;
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
    densityHU::Array{Float64,3};
    waterEq::Bool

    # rank
    r::Int; #for adaptive this is initial rank
    rMax::Int; #and this is the max allowed rank

    gridSize::Array{Int,1};
    gridWidth::Array{Float64,1};
    gridScale::Float64; #scale factor of grid refinement relative to that given from an initial CT

    # tolerance for rank adaptivity
    epsAdapt::Float64;  
    adaptIndex::Float64;

    #file names
    model::String;

    order::Int64

    function Settings(problem::String, model::String,Nx::Int64,Ny::Int64,Nz::Int64, nPN::Int, r::Int, epsAdapt::Float64, particle::String, order::Int,gridScale::Float64=1.0)        
        #Proton rest energy
        if particle == "Protons"
            eRest = 938.26 #MeV
        elseif particle == "Electrons"
            eRest = 0.5 #MeV -> estimate, look this up
        end
        # spatial grid setting
        if order ==1
            NCellsX = Nx - 1;
            NCellsY = Ny - 1;
            NCellsZ = Nz - 1;
        elseif order == 2
            NCellsX = Nx - 3;
            NCellsY = Ny - 3;
            NCellsZ = Nz - 3;
        end

        a = 0.0; # left boundary
        b = 14.5; # right boundary

        c = 0.0; # lower boundary
        d = 14.5; # upper boundary

        e = 0.0; # left z boundary
        f = 14.5; # right z boundary

        density = ones(NCellsX,NCellsY,NCellsZ); 
        densityHU = zeros(NCellsX,NCellsY,NCellsZ); #HU

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0;
        eMax = 1.0;
        rMax = r;
        x0 = 0.5*b;
        y0 = 0.5*d;
        z0 = 1.0*f;
        Omega1 = 0.0;
        Omega2 = 0.0;
        Omega3 = 1.0;
        if model == "Boltzmann"
            OmegaMin = 0;
        else 
            OmegaMin=180;
        end
        densityMin = 0.2;
        adaptIndex = 0;
        sigmaX = 0.3;
        sigmaY = 0.3;
        sigmaZ = 0.01;
        eMin = 0.001;
        gridSize = [NCellsX,NCellsY,NCellsZ]; 

        nE_perBeam = [0; 1]
        waterEq = false
        if problem == "BoxInsert"
            a = 0; # left boundary
            b = 2; # right boundary
            c = 0; # lower boundary
            d = 2; # upper boundary
            e = 0;
            f = 7;
            if particle == "Protons"
                mu_e = 80
            else
                mu_e =5
            end
            w_e = 1;
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e #+ 5*sigmaE; 
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            #sigmaZ = sqrt((0.0022*1.77*(eKin^0.77))^2*(sigmaE*eKin)); #approx to energy distr. only for protons
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = 0.0;
            Omega2 = 0.0;
            Omega3 = 1.0;
            x0 = 0.5 * b;
            y0 = 0.5 * d;
            z0 = 0.01 * f;
            #sizeOfTracerCT = [40,40,160];
            density = density.*1.018 #value like tracer
            if particle == "Protons"
                density[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.3))+1:Int(floor(NCellsZ*0.6))] .= 0.6190303991130821; #inserted box of lower density 
                densityHU[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.3))+1:Int(floor(NCellsZ*0.6))] .= -400; #inserted box of lower density defined in HU
            else
                density[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.1))+1:Int(floor(NCellsZ*0.3))] .= 0.6190303991130821; #inserted box of lower density 
                densityHU[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.1))+1:Int(floor(NCellsZ*0.3))] .= -400; #inserted box of lower density defined in HU
            end
        elseif problem == "WaterPhantom"
            nB=1;
            a = 0; # left boundary
            b = 4; # right boundary
            c = 0; # lower boundary
            d = 4; # upper boundary
            e = 0;
            f = 4;
            if particle == "Protons"
                mu_e = 60
            else
                mu_e =5
            end
            w_e = 1;
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e #+ 5*sigmaE; 
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            #sigmaZ = sqrt((0.0022*1.77*(eKin^0.77))^2*(sigmaE*eKin));  #approx to energy distr.
            sigmaZ = 0.1; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = 0.0;
            Omega2 = 0.0;
            Omega3 = 1.0;
            x0 = 0.5 * b;
            y0 = 0.5 * d;
            z0 = 0.01* f;
        elseif problem == "dicomImport"
            Omega1 = [0.0,0.0];
            Omega2 = [1.0,-1.0];
            Omega3 = [0.0,0.0];
            nB = size(Omega1)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)
            #read dicom file
            densityHU, res = load_ct_volume(dataFile)
            NCellsX = size(densityHU,1)
            NCellsY = size(densityHU,2)
            NCellsZ = size(densityHU,3)

            dx = res[1]/10 # divide by 10 bc of unit conversion mm -> cm
            dy = res[2]/10
            dz = res[3]/10
            a = 0.0; # left boundary
            b = NCellsX * dx; # right boundary
            c = 0.0; # lower boundary
            d = NCellsY * dy; # upper boundary
            e = 0.0;
            f = NCellsZ * dz;
            density = reshape(HUtoDensity(densityHU[:]),NCellsX,NCellsY,NCellsZ)

            w_e = [1, 1];
             if particle == "Protons"
                mu_e = [165, 165]
            else
                mu_e = [10,10]
            end
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = maximum(eKin) + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            x0[1] = 0.55 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.55 * f;
            x0[2] = 0.55 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.55 * f;

            #add option for finer grid for dose comp (maybe also only for DLRA)
            if gridScale[1] != 1.0 || gridScale[2] != 1.0  || gridScale[3] != 1.0 
                gridCTtoDose = interpolate((collect(range(a,b,size(density,1))),collect(range(c,d,size(density,2))),collect(range(e,f,size(density,3)))), density,Gridded(Linear()))
                gridCTtoDoseHU = interpolate((collect(range(a,b,size(density,1))),collect(range(c,d,size(density,2))),collect(range(e,f,size(density,3)))), densityHU,Gridded(Linear()))
            
                density = gridCTtoDose(collect(range(a,b,Int(size(density,1)*gridScale[1]))),collect(range(c,d,Int(size(density,2)*gridScale[2]))),collect(range(e,f,Int(size(density,3)*gridScale[3]))))
                densityHU = gridCTtoDoseHU(collect(range(a,b,Int(size(densityHU,1)*gridScale[1]))),collect(range(c,d,Int(size(densityHU,2)*gridScale[2]))),collect(range(e,f,Int(size(densityHU,3)*gridScale[3]))))
                
                NCellsX = size(density,1)
                NCellsY = size(density,2)
                NCellsZ = size(density,3)
                dx = dx/gridScale[1]
                dy = dy/gridScale[2]
                dz = dz/gridScale[3]
            else
                NCellsX = size(density,1)
                NCellsY = size(density,2)
                NCellsZ = size(density,3)
                density = density[idx[1],idx[2],idx[3]]
                densityHU = densityHU[idx[1],idx[2],idx[3]]
            end
            #crop away air at the boundaries and regions far away from beam #
            density, idx=trim_density(density,eps=0.1,beams=[(SVector(x0[1], y0[1],z0[1]), SVector(3*sigmaX,3*sigmaY,sigmaE[1]), 100.0,SVector(Omega1[1], Omega2[1], Omega3[1])),(SVector(x0[2], y0[2],z0[2]), SVector(3*sigmaX,3*sigmaY,sigmaE[2]), 100.0, SVector(Omega1[2], Omega2[2], Omega3[2]))],x_range=(a,b), y_range=(c,d), z_range=(e,f))
            densityHU = densityHU[idx[1],idx[2],idx[3]]
            
            NCellsX = size(density,1)
            NCellsY = size(density,2)
            NCellsZ = size(density,3)
            a = 0.0; # left boundary
            b = NCellsX * dx; # right boundary, divide by 10 bc of unit conversion mm -> cm
            c = 0.0; # lower boundary
            d = NCellsY * dy; # upper boundary
            e = 0.0;
            f = NCellsZ * dz;

            if order ==1
                Nx = NCellsX + 1;
                Ny = NCellsY + 1;
                Nz = NCellsZ + 1;
            elseif order == 2
                Nx = NCellsX + 3;
                Ny = NCellsY + 3;
                Nz = NCellsZ + 3;
            end

            #redefine beam position because of changed grid, this probably needs to be done smarter (actually find equivalent position in changed grid or keep beam pos and change box borders (a,b),...)
            x0[1] = 0.5 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.5 * f;
            x0[2] = 0.5 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.5 * f;

            #reduce to only one beam
            # x0 = 0.5 * b;
            # y0 = 0.0 * d;
            # z0 = 0.5 * f;
            # x0 = 0.5 * b;
            # y0 = 1.0 * d;
            # z0 = 0.5 * f;
            # Omega1 = Omega1[1]
            # Omega2 = Omega2[1]
            # Omega3= Omega3[1]
            # Omega1 = Omega1[2]
            # Omega2 = Omega2[2]
            # Omega3 = Omega3[2]
            # nB = 1
            #Q is a limit for minimum density needed for stability?
        elseif problem == "matImport"
            Omega1 = [0.0,0.0];
            Omega2 = [1.0,-1.0];
            Omega3 = [0.0,0.0];
            nB = size(Omega1)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)
            #read matfile (expected to be in the style as matRad phantoms)
            dataFile = "testCase_CSD/data/PROSTATE.mat"
            file = matopen(dataFile, "r")
            tmp = read(file, "ct") # note that this does NOT introduce a variable ``varname`` into scope
            # spatial grid setting
            NCellsX = tmp["cubeDim"][1]
            NCellsY = tmp["cubeDim"][2]
            NCellsZ = tmp["cubeDim"][3]

            dx = tmp["resolution"]["x"]/10/4 # divide by 10 bc of unit conversion mm -> cm, divide by 4 for smaller test case
            dy = tmp["resolution"]["y"]/10/4
            dz = tmp["resolution"]["z"]/10/4
            a = 0.0; # left boundary
            b = NCellsX * dx/2; # right boundary
            c = 0.0; # lower boundary
            d = NCellsY * dy/2; # upper boundary
            e = 0.0;
            f = NCellsZ * dz/2;

            density =  tmp["cube"][1]
            densityHU =  tmp["cubeHU"][1]
            tmp = nothing
            close(file)
            w_e = [1, 1];
             if particle == "Protons"
                mu_e = [60, 60]
            else
                mu_e = [5,5]
            end
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = maximum(eKin) + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            #sigmaZ = sqrt((0.0022*1.77*(eKin^0.77))^2*(sigmaE*eKin));
            sigmaZ = 0.01; 
            x0[1] = 0.55 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.55 * f;
            x0[2] = 0.55 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.55 * f;
            #crop away air at the boundaries and regions far away from beam #
            density_cropped, idx=trim_density(density,eps=0.1,beams=[(SVector(x0[1], y0[1],z0[1]), SVector(sigmaX*3,sigmaY*3,sigmaE[1]), 100.0,SVector(Omega1[1], Omega2[1], Omega3[1]))],x_range=(a,b), y_range=(c,d), z_range=(e,f))
            densityHU_cropped = densityHU[idx[1],idx[2],idx[3]]

            #add option for finer grid for dose comp (maybe also only for DLRA)
            if gridScale != 1.0
                gridCTtoDose = interpolate((collect(range(a,b,size(density_cropped,1))),collect(range(c,d,size(density_cropped,2))),collect(range(e,f,size(density_cropped,3)))), density_cropped,Gridded(Linear()))
                gridCTtoDoseHU = interpolate((collect(range(a,b,size(density_cropped,1))),collect(range(c,d,size(density_cropped,2))),collect(range(e,f,size(density_cropped,3)))), densityHU_cropped,Gridded(Linear()))
            
                density = gridCTtoDose(collect(range(a,b,Int(round.(size(density_cropped,1)*gridScale)))),collect(range(c,d,Int(round.(size(density_cropped,2)*gridScale)))),collect(range(e,f,Int(round.(size(density_cropped,3)*gridScale)))))
                densityHU = gridCTtoDoseHU(collect(range(a,b,Int(round.(size(density_cropped,1)*gridScale)))),collect(range(c,d,Int(round.(size(density_cropped,2)*gridScale)))),collect(range(e,f,Int(round.(size(density_cropped,3)*gridScale)))))
                
                NCellsX = Int(round.(size(density_cropped,1)*gridScale))
                NCellsY = Int(round.(size(density_cropped,2)*gridScale))
                NCellsZ = Int(round.(size(density_cropped,3)*gridScale))
                dx = dx/gridScale
                dy = dy/gridScale
                dz = dz/gridScale
            else
                NCellsX = size(density_cropped,1)
                NCellsY = size(density_cropped,2)
                NCellsZ = size(density_cropped,3)
                density = density[idx[1],idx[2],idx[3]]
                densityHU = densityHU[idx[1],idx[2],idx[3]]
            end
            
            a = 0.0; # left boundary
            b = NCellsX * dx/2; # right boundary, divide by 10 bc of unit conversion mm -> cm
            c = 0.0; # lower boundary
            d = NCellsY * dy/2; # upper boundary
            e = 0.0;
            f = NCellsZ * dz/2;

            if order ==1
                Nx = NCellsX + 1;
                Ny = NCellsY + 1;
                Nz = NCellsZ + 1;
            elseif order == 2
                Nx = NCellsX + 3;
                Ny = NCellsY + 3;
                Nz = NCellsZ + 3;
            end
            
            #redefine beam position because of changed grid, this probably needs to be done smarter (actually find equivalent position in changed grid or keep beam pos and change box borders (a,b),...)
            x0[1] = 0.5 * b;
            y0[1] = 0.0 * d;
            z0[1] = 0.5 * f;
            x0[2] = 0.5 * b;
            y0[2] = 1.0 * d;
            z0[2] = 0.5 * f;

            #reduce to only one beam
            x0 = 0.5 * b;
            y0 = 0.0 * d;
            z0 = 0.5 * f;
            # x0 = 0.5 * b;
            # y0 = 1.0 * d;
            # z0 = 0.5 * f;
            Omega1 = Omega1[1]
            Omega2 = Omega2[1]
            Omega3= Omega3[1]
            # Omega1 = Omega1[2]
            # Omega2 = Omega2[2]
            # Omega3 = Omega3[2]
            nB = 1
            #Q is a limit for minimum density needed for stability?
        end
        sigmaT = sigmaA + sigmaS;

        # spatial grid
        x = collect(range(a,stop = b,length = NCellsX));
        if order == 2
            # Initialize the grid with ghost cells for second-order accuracy
            x = collect(range(a, stop=b, length=NCellsX))
            dx = x[2] - x[1]
            y = collect(range(c, stop=d, length=NCellsY))
            dy = y[2] - y[1]
            z = collect(range(e, stop=f, length=NCellsZ))
            dz = z[2]-z[1];

            # Add two ghost cells on each boundary
            x = [x[1] - 2*dx; x[1] - dx; x; x[end] + dx]
            y = [y[1] - 2*dy; y[1] - dy; y; y[end] + dy]
            z = [z[1] - 2*dz; z[1] - dz; z; z[end] + dz]

            # Calculate the cell boundaries by shifting by half a grid spacing
            x = x .+ dx/2
            y = y .+ dy/2
            z = z .+ dz/2

            # Calculate the midpoints of the cells
            xMid = x[2:(end-2)] .+ 0.5 * dx
            yMid = y[2:(end-2)] .+ 0.5 * dy
            zMid = z[2:(end-2)] .+ 0.5 * dz
        else
            x = collect(midpoints(a, b, NCellsX)) #collect(range(a, stop=b, length=NCellsX))
            dx = x[2] - x[1]
            y = collect(midpoints(c, d, NCellsY)) #collect(range(c, stop=d, length=NCellsY))
            dy = y[2] - y[1]
            z = collect(midpoints(e, f, NCellsZ)) #collect(range(e, stop=f, length=NCellsZ))
            dz = z[2]-z[1];
            x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
            x = x.+dx/2;
            xMid = collect(midpoints(a, b, NCellsX)) #collect(range(a, stop=b, length=NCellsX))
            y = [y[1]-dy;y]; # add ghost cells so that boundary cell centers lie on a and b
            y = y.+dy/2;
            yMid = collect(midpoints(c, d, NCellsY)) #collect(range(c, stop=d, length=NCellsY))
            z = [z[1]-dz;z]; # add ghost cells so that boundary cell centers lie on a and b
            z = z.+dz/2;
            zMid = collect(midpoints(e, f, NCellsZ)) #collect(range(e, stop=f, length=NCellsZ))
        end
        gridWidth = [dx,dy,dz]; 
        # time settings
        if particle == "Electrons"
            cfl = 0.1 * minimum(mu_e) *minimum(density);# CFL condition
        else
            cfl = 0.6 * minimum(mu_e) #*minimum(density)# CFL condition
        end
        println("cfl = $cfl")
        dE = cfl*min(dx,dy,dz)#*minimum(density);
        sigmaE = maximum(sigmaE)
        # build class
        new(Nx,Ny,Nz,NCellsX,NCellsY,NCellsZ,a,b,c,d,e,f,dx,dy,dz,eMax,eMin,eRest,dE,cfl,N_E,nPN,x,xMid,y,yMid,z,zMid,problem,particle,x0,y0,z0,Omega1,Omega2,Omega3,OmegaMin,densityMin,sigmaX,sigmaY,sigmaZ,sigmaE,sigmaT,sigmaS,density,densityHU,waterEq,r,rMax,gridSize,gridWidth,gridScale,epsAdapt,adaptIndex,model,order);
    end
end

function midpoints(a, b, nx)
    h = (b - a) / nx
    return a .+ (0:nx-1) .* h .+ h/2
end