__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum
using TensorToolbox

include("CSD.jl")
include("PNSystem.jl")
include("quadratures/Quadrature.jl")
include("utils.jl")

mutable struct SolverCSD
    # spatial grid of cell interfaces
    x::Array{Float64};
    y::Array{Float64};
    xGrid::Array{Float64,2};
    xi::Array{Float64,1};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # Roe matrix
    AbsAx::SparseMatrixCSC{Float64, Int64};
    AbsAz::SparseMatrixCSC{Float64, Int64};
    AxPlus::Array{Float64,2};
    AxMinus::Array{Float64,2};
    AzPlus::Array{Float64,2};
    AzMinus::Array{Float64,2};
    # normalized Legendre Polynomials
    P::Array{Float64,2};
    # quadrature points
    mu::Array{Float64,1};
    w::Array{Float64,1};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # dose vector
    dose::Array{Float64,1};

    L1x::SparseMatrixCSC{Float64, Int64};
    L1y::SparseMatrixCSC{Float64, Int64};
    L2x::SparseMatrixCSC{Float64, Int64};
    L2y::SparseMatrixCSC{Float64, Int64};
    boundaryIdx::Array{Int,1}
    boundaryBeam::Array{Int,1}

    Q::Quadrature
    O::Array{Float64,2};
    M::Array{Float64,2};

    OReduced::Array{Float64,2};
    MReduced::Array{Float64,2};
    qReduced::Array{Float64,2};

    # constructor
    function SolverCSD(settings)
        x = settings.x;
        y = settings.y;

        # setup flux matrix
        gamma = zeros(settings.nPN+1);
        for i = 1:settings.nPN+1
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        A = zeros(settings.nPN,settings.nPN);
            # setup flux matrix (alternative analytic computation)
        for i = 1:(settings.nPN-1)
            n = i-1;
            A[i,i+1] = (n+1)/(2*n+1)*sqrt(gamma[i+1])/sqrt(gamma[i]);
        end

        for i = 2:settings.nPN
            n = i-1;
            A[i,i-1] = n/(2*n+1)*sqrt(gamma[i-1])/sqrt(gamma[i]);
        end

        # construct CSD fields
        csd = CSD(settings);

        # construct PN system matrices
        pn = PNSystem(settings)
        Ax,Ay,Az = SetupSystemMatrices(pn);
        SetupSystemMatricesSparse(pn);

        # setup Roe matrix
        S = eigvals(Ax)
        V = eigvecs(Ax)
        AbsAx = V*abs.(diagm(S))*inv(V)
        idxPos = findall((S.>=0.0))
        idxNeg = findall((S.<0.0))
        SPlus = zeros(size(S)); SPlus[idxPos] = S[idxPos];
        SMinus = zeros(size(S)); SMinus[idxNeg] = S[idxNeg];
        AxPlus = V*diagm(SPlus)*inv(V)
        AxMinus = V*diagm(SMinus)*inv(V)

        idx = findall(abs.(AbsAx) .> 1e-10)
        Ix = first.(Tuple.(idx)); Jx = last.(Tuple.(idx)); vals = AbsAx[idx];
        AbsAx = sparse(Ix,Jx,vals,pn.nTotalEntries,pn.nTotalEntries);
        
        S = eigvals(Az)
        V = eigvecs(Az)
        AbsAz = V*abs.(diagm(S))*inv(V)
        idxPos = findall((S.>=0.0))
        idxNeg = findall((S.<0.0))
        SPlus = zeros(size(S)); SPlus[idxPos] = S[idxPos];
        SMinus = zeros(size(S)); SMinus[idxNeg] = S[idxNeg];
        AzPlus = V*diagm(SPlus)*inv(V)
        AzMinus = V*diagm(SMinus)*inv(V)

        idx = findall(abs.(AbsAz) .> 1e-10)
        Iz = first.(Tuple.(idx)); Jz = last.(Tuple.(idx)); valsz = AbsAz[idx];
        AbsAz = sparse(Iz,Jz,valsz,pn.nTotalEntries,pn.nTotalEntries);

        # allocate dose vector
        dose = zeros(settings.NCellsX*settings.NCellsY)

        # compute normalized Legendre Polynomials
        Nq=200;
        (mu,w) = gauss(Nq);
        P=zeros(Nq,settings.nPN);
        for k=1:Nq
            PCurrent = collectPl(mu[k],lmax=settings.nPN-1);
            for i = 1:settings.nPN
                P[k,i] = PCurrent[i-1]/sqrt(gamma[i]);
            end
        end

        # setupt stencil matrix
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        N = pn.nTotalEntries;
        L1x = spzeros(nx*ny,nx*ny);
        L1y = spzeros(nx*ny,nx*ny);
        L2x = spzeros(nx*ny,nx*ny);
        L2y = spzeros(nx*ny,nx*ny);

        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3*(nx-2)*(ny-2)); J = zeros(3*(nx-2)*(ny-2)); vals = zeros(3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # x part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i+1,j);
                indexMinus = vectorIndex(ny,i-1,j);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dx; 
                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx;
                end
                if i < nx
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dx; 
                end
            end
        end
        L1x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(3*(nx-2)*(ny-2)); J .= zeros(3*(nx-2)*(ny-2)); vals .= zeros(3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # y part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i,j+1);
                indexMinus = vectorIndex(ny,i,j-1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dy; 

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy;
                end
                if j < ny
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dy; 
                end
            end
        end
        L1y = sparse(II,J,vals,nx*ny,nx*ny);

        II = zeros(2*(nx-2)*(ny-2)); J = zeros(2*(nx-2)*(ny-2)); vals = zeros(2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # x part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i+1,j);
                indexMinus = vectorIndex(ny,i-1,j);

                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx;
                end
                if i < nx
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dx;
                end
            end
        end
        L2x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(2*(nx-2)*(ny-2)); J .= zeros(2*(nx-2)*(ny-2)); vals .= zeros(2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # y part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i,j+1);
                indexMinus = vectorIndex(ny,i,j-1);

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy;
                end
                if j < ny
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dy;
                end
            end
        end
        L2y = sparse(II,J,vals,nx*ny,nx*ny);

        # collect boundary indices (double check this)
        boundaryIdx = zeros(Int,2*nx+2*ny)
        counter = 0;
        for i = 1:nx
            counter +=1;
            j = 1;
            idx = (i-1)*ny + j;
            boundaryIdx[counter] = idx
            counter +=1;
            j = ny;
            idx = (i-1)*ny + j;
            boundaryIdx[counter] = idx
        end

        for j = 1:ny
            counter +=1;
            i = 1;
            idx = (i-1)*ny + j;
            boundaryIdx[counter] = idx
            counter +=1;
            i = nx;
            idx = (i-1)*ny + j;
            boundaryIdx[counter] = idx
        end

        boundaryBeam = zeros(Int,2*nx) # boundary indices uncollided particles for beam
        counter = 0;
        for i = 1:nx
            counter += 1;
            j = 1;
            idx = (i-1)*ny + j;
            boundaryBeam[counter] = idx
            counter += 1;
            j = 2;
            idx = (i-1)*ny + j;
            boundaryBeam[counter] = idx
        end

        # setup spatial grid
        xGrid = zeros(nx*ny,2)
        for i = 1:nx
            for j = 1:ny
                # y part
                index = vectorIndex(ny,i,j);
                xGrid[index,1] = settings.xMid[i];
                xGrid[index,2] = settings.yMid[j];
            end
        end

        # setup quadrature
        qorder = settings.nPN+1; # must be even for standard quadrature
        qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
        Q = Quadrature(qorder,qtype)

        weights = Q.weights
        #Norder = (qorder+1)*(qorder+1)
        Norder = pn.nTotalEntries
        nq = length(weights);

        # Construct Gauss quadrature
        mu,gaussweights = gausslegendre(qorder)
            
        # around z axis equidistant
        phi = [(k+0.5)*pi/qorder for k=0:2*qorder-1]

        # Transform between (mu,phi) and (x,y,z)
        x = sqrt.(1.0 .- mu.^2).*cos.(phi)'
        y = sqrt.(1.0 .- mu.^2).*sin.(phi)'
        z =           mu    .*ones(size(phi))'
        weights = 2.0*pi/qorder*repeat(gaussweights,1,2*qorder)
            
        weights = weights[:]*0.5;

    
        global counter;
        counter = 1
        O = zeros(nq,Norder)
        M = zeros(Norder,nq)
        for l=0:settings.nPN
            for m=-l:l
                for k = 1:length(mu)
                    for j = 1:length(phi)
                        global counter;
                        O[(j-1)*qorder+k,counter] =  real_sph(mu[k],phi[j],l,m)
                        M[counter,(j-1)*qorder+k] = O[(j-1)*qorder+k,counter]*weights[(j-1)*qorder+k]
                    end
                end
                counter += 1
            end
        end

        xi, w = gausslegendre(settings.Nxi);

        new(x,y,xGrid,xi,settings,gamma,AbsAx,AbsAz,AxPlus,AxMinus,AzPlus,AzMinus,P,mu,w,csd,pn,dose,L1x,L1y,L2x,L2y,boundaryIdx,boundaryBeam,Q,O,M);
    end
end

function SetupIC(obj::SolverCSD)
    nq = obj.Q.nquadpoints;
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    psi = zeros(obj.settings.NCellsX,obj.settings.NCellsY,nq);

    if obj.settings.problem == "validation"
        for i = 1:nx
            for j = 1:ny
                for k = 1:nq
                    sigmaO1Inv = 100.0;
                    sigmaO3Inv = 100.0;
                    pos_beam = [obj.settings.x0,obj.settings.y0,0];
                    space_beam = normpdf(obj.settings.xMid[i],pos_beam[1],.1).*normpdf(obj.settings.yMid[j],pos_beam[2],.1);
                    trafo = 1.0;#obj.csd.S[1]*obj.settings.density[i,j];
                    psi[i,j,k] = 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-obj.Q.pointsxyz[k,1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-obj.Q.pointsxyz[k,3])^2)*space_beam*trafo;
                end
            end
        end
    elseif obj.settings.problem == "lung2"
        sigma = 0.1;
        OmegaStar = [obj.settings.Omega1,0.0,obj.settings.Omega3]
        pos_beam = [obj.settings.x0;obj.settings.y0]
        for i = 1:nx
            for j = 1:ny
                space_beam = normpdf(obj.settings.xMid[i],pos_beam[1],sigma).*normpdf(obj.settings.yMid[j],pos_beam[2],sigma);
                for k = 1:nq
                    Omega = obj.Q.pointsxyz[k,:];
                    Omega_beam = normpdf(Omega[1],OmegaStar[1],sigma).*normpdf(Omega[2],OmegaStar[2],sigma).*normpdf(Omega[3],OmegaStar[3],sigma);
                    psi[i,j,k] = 10^5*Omega_beam*space_beam*obj.csd.S[1]*obj.settings.density[i,j];
                end
            end
        end
    else    
        for k = 1:nq
            psi[:,:,k] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)
        end
    end
    
    return psi;
end

function SetupICMoments(obj::SolverCSD)
    u = zeros(obj.settings.NCellsX,obj.settings.NCellsY,obj.pn.nTotalEntries);
    
    if obj.settings.problem == "CT"
        PCurrent = collectPl(1,lmax=obj.settings.nPN);
        for l = 0:obj.settings.nPN
            for k=-l:l
                i = GlobalIndex( l, k )+1;
                #u[:,:,i] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)*Y[i];#obj.csd.StarMAPmoments[i]# .* PCurrent[l]/sqrt(obj.gamma[l+1])
                u[:,:,i] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)*obj.csd.StarMAPmoments[i]
            end
        end
    elseif obj.settings.problem == "validation"
        nq = obj.Q.nquadpoints;
        nx = obj.settings.NCellsX;
        ny = obj.settings.NCellsY;
        @polyvar xx yy zz
        phi_beam = pi/2;                               # Angle of beam w.r.t. x-axis.
        mu_beam = 0;  
        psi = zeros(obj.pn.nTotalEntries)*1im;
        counter = 1;
        for l=0:obj.settings.nPN
            for m=-l:l
                sphericalh = ylm(l,m,xx,yy,zz)
                psi[counter] =  sph_cc(mu_beam,phi_beam,l,m)
                counter += 1
            end
        end
    
        for i = 1:nx
            for j = 1:ny
                pos_beam = [0.0,0.5*14.5,0];
                space_beam = normpdf(obj.settings.xMid[i],pos_beam[1],.01).*normpdf(obj.settings.yMid[j],pos_beam[2],.01);
                trafo = 1.0;#obj.csd.S[1]*obj.settings.density[i,j];
                u[i,j,:] = Float64.(obj.pn.M*psi)*space_beam;
            end
        end
    elseif obj.settings.problem == "2D" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD"
        for l = 0:obj.settings.nPN
            for k=-l:l
                i = GlobalIndex( l, k )+1;
                u[:,:,i] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)*obj.csd.StarMAPmoments[i]
            end
        end
    end
    return u;
end

function PsiLeft(obj::SolverCSD,n::Int,mu::Float64)
    E0 = obj.settings.eMax;
    return 10^5*exp(-200.0*(1.0-mu)^2)*exp(-50*(E0-E)^2)
end

function PsiBeam(obj::SolverCSD,Omega::Array{Float64,1},E::Float64,x::Float64,y::Float64,n::Int)
    E0 = obj.settings.eMax;
    if obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "timeCT"
        sigmaO1Inv = 300.0;
        sigmaO3Inv = 300.0;
        sigmaXInv = 50.0;
        sigmaYInv = 50.0;
        sigmaEInv = 100.0;
        OmegaStar = [obj.settings.Omega1; 0.0; obj.settings.Omega3]
    elseif obj.settings.problem == "liver"
        sigmaO1Inv = 75.0;
        sigmaO3Inv = 0.0;
        sigmaXInv = 10.0;
        sigmaYInv = 10.0;
        sigmaEInv = 10.0;
    elseif obj.settings.problem == "validation"
        sigmaO1Inv = 10000.0;
        sigmaO3Inv = 10000.0;
        densityMin = 1.0;
        pos_beam = [obj.settings.x0,obj.settings.y0,0];
        space_beam = normpdf(x,pos_beam[1],.1).*normpdf(y,pos_beam[2],.1);
        #println(space_beam)
        return 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2)*space_beam*obj.csd.S[n]*densityMin;
    elseif obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2"
        return 0.0;
    end

    return 10^5*exp(-sigmaO1Inv*(norm(OmegaStar-Omega)^2))*exp(-sigmaEInv*(E0-E)^2)*exp(-sigmaXInv*(x-obj.settings.x0)^2)*exp(-sigmaYInv*(y-obj.settings.y0)^2)*obj.csd.S[n]*obj.settings.densityMin;
end

function PsiBeam(obj::SolverCSD,Omega::Array{Float64,1},E::Float64,x::Float64,y::Float64,xi::Float64,n::Int)
    E0 = obj.settings.eMax;
    if obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "timeCT"
        sigmaO1Inv = 300.0;
        sigmaO3Inv = 300.0;
        sigmaXInv = 50.0;
        sigmaYInv = 50.0;
        sigmaEInv = 100.0;
        OmegaStar = [obj.settings.Omega1; 0.0; obj.settings.Omega3]
    elseif obj.settings.problem == "liver"
        sigmaO1Inv = 75.0;
        sigmaO3Inv = 0.0;
        sigmaXInv = 10.0;
        sigmaYInv = 10.0;
        sigmaEInv = 10.0;
    elseif obj.settings.problem == "validation"
        sigmaO1Inv = 10000.0;
        sigmaO3Inv = 10000.0;
        densityMin = 1.0;
        pos_beam = [0.5*14.5,0.0,0];
        space_beam = normpdf(x,pos_beam[1],.01).*normpdf(y,pos_beam[2],.01);
        #println(space_beam)
        return 10^5*exp(-sigmaO1Inv*(norm(OmegaStar-Omega)))*space_beam*obj.csd.S[n]*densityMin;
    elseif obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD"
        return 0.0;
    end
    #return 10^5*exp(-sigmaO1Inv*(norm(OmegaStar-Omega)^2))*exp(-sigmaEInv*(E0-E)^2)*exp(-sigmaXInv*(x-obj.settings.x0 + xi)^2)*exp(-sigmaYInv*(y-obj.settings.y0 + xi)^2)*obj.csd.S[n]*obj.settings.densityMin;
    return 10^5*exp(-sigmaO1Inv*(norm(OmegaStar-Omega)^2))*exp(-sigmaEInv*(E0-E)^2)*exp(-sigmaXInv*(x-obj.settings.x0)^2)*exp(-sigmaYInv*(y-obj.settings.y0)^2)*obj.csd.S[n]*obj.settings.densityMin;
end

function BCLeft(obj::SolverCSD,n::Int)
    if obj.settings.problem == "WaterPhantomEdgar"
        E0 = obj.settings.eMax;
        E = obj.csd.eGrid[n];
        PsiLeft = 10^5*exp.(-200.0*(1.0.-obj.mu).^2)*exp(-50*(E0-E)^2)
        uHat = zeros(obj.settings.nPN)
        for i = 1:obj.settings.nPN
            uHat[i] = sum(PsiLeft.*obj.w.*obj.P[:,i]);
        end
        return uHat*obj.density[1]*obj.csd.S[n]
    else
        return 0.0
    end
end


function solveFluxUpwind!(obj::SolverCSD, phi::Array{Float64,3}, flux::Array{Float64,3})
    # computes the numerical flux over cell boundaries for each ordinate
    # for faster computation, we split the iteration over quadrature points
    # into four different blocks: North West, Nort East, Sout West, South East
    # this corresponds to the direction the ordinates point to
    idxPosPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxPosNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxNegPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,3].>=0.0))
    idxNegNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,3].<0.0))

    nx = collect(2:(obj.settings.NCellsX-1));
    ny = collect(2:(obj.settings.NCellsY-1));

    # PosPos
    for j=ny,i=nx, q = idxPosPos
        s2 = phi[i,j-1,q]
        s3 = phi[i,j,q]
        northflux = s3
        southflux = s2

        s2 = phi[i-1,j,q]
        s3 = phi[i,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,3]./obj.settings.dy .* (northflux-southflux)
    end
    #PosNeg
    for j=ny,i=nx,q = idxPosNeg
        s2 = phi[i,j,q]
        s3 = phi[i,j+1,q]
        northflux = s3
        southflux = s2

        s2 = phi[i-1,j,q]
        s3 = phi[i,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegPos
    for j=ny,i=nx,q = idxNegPos
        s2 = phi[i,j-1,q]
        s3 = phi[i,j,q]
        northflux = s3
        southflux = s2

        s2 = phi[i,j,q]
        s3 = phi[i+1,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegNeg
    for j=ny,i=nx,q = idxNegNeg
        s2 = phi[i,j,q]
        s3 = phi[i,j+1,q]
        northflux = s3
        southflux = s2

        s2 = phi[i,j,q]
        s3 = phi[i+1,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end
end

function solveFluxUpwind!(obj::SolverCSD, phi::Array{Float64,4}, flux::Array{Float64,4})
    # computes the numerical flux over cell boundaries for each ordinate
    # for faster computation, we split the iteration over quadrature points
    # into four different blocks: North West, Nort East, Sout West, South East
    # this corresponds to the direction the ordinates point to
    idxPosPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxPosNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxNegPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,3].>=0.0))
    idxNegNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,3].<0.0))

    nx = collect(2:(obj.settings.NCellsX-1));
    ny = collect(2:(obj.settings.NCellsY-1));
    nxi = collect(1:(obj.settings.Nxi));

    rho0Inv = obj.settings.rho0Inv;
    rho1Inv = obj.settings.rho1Inv;

    # PosPos
    for j=ny,i=nx,l=nxi, q = idxPosPos
        s2 = (rho0Inv[i,j-1]+obj.xi[l]*rho1Inv[i,j-1])*phi[i,j-1,q,l]
        s3 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        northflux = s3
        southflux = s2

        s2 = (rho0Inv[i-1,j]+obj.xi[l]*rho1Inv[i-1,j])*phi[i-1,j,q,l]
        s3 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        eastflux = s3
        westflux = s2

        flux[i,j,q,l] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,3]./obj.settings.dy .* (northflux-southflux)
    end
    #PosNeg
    for j=ny,i=nx,l=nxi,q = idxPosNeg
        s2 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        s3 = (rho0Inv[i,j+1]+obj.xi[l]*rho1Inv[i,j+1])*phi[i,j+1,q,l]
        northflux = s3
        southflux = s2

        s2 = (rho0Inv[i-1,j]+obj.xi[l]*rho1Inv[i-1,j])*phi[i-1,j,q,l]
        s3 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        eastflux = s3
        westflux = s2

        flux[i,j,q,l] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegPos
    for j=ny,i=nx,l=nxi,q = idxNegPos
        s2 = (rho0Inv[i,j-1]+obj.xi[l]*rho1Inv[i,j-1])*phi[i,j-1,q,l]
        s3 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        northflux = s3
        southflux = s2

        s2 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        s3 = (rho0Inv[i+1,j]+obj.xi[l]*rho1Inv[i+1,j])*phi[i+1,j,q,l]
        eastflux = s3
        westflux = s2

        flux[i,j,q,l] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegNeg
    for j=ny,i=nx,l=nxi,q = idxNegNeg
        s2 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        s3 = (rho0Inv[i,j+1]+obj.xi[l]*rho1Inv[i,j+1])*phi[i,j+1,q,l]
        northflux = s3
        southflux = s2

        s2 = (rho0Inv[i,j]+obj.xi[l]*rho1Inv[i,j])*phi[i,j,q,l]
        s3 = phi[i+1,j,q,l]
        eastflux = s3
        westflux = s2

        flux[i,j,q,l] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end
end

function SolveFirstCollisionSource(obj::SolverCSD,xi::Float64=0.0)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries

    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);
    rho1InvXi = Diagonal(s.rho1InvVec).*xi;

    # Set up initial condition and store as matrix
    psi = SetupIC(obj);
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam]
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.MReduced = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));
    density = (s.rho0Inv .+ s.rho1Inv.*xi).^(-1);
    densityVec = (s.rho0InvVec .+ s.rho1InvVec.*xi).^(-1);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE


    u = zeros(nx*ny,N);
    uNew = deepcopy(u)
    flux = zeros(size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(nx*ny);

    psiNew = zeros(size(psi));
    uTilde = zeros(size(u))

    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if obj.settings.problem != "validation" # validation testcase sets beam in initial condition
            for k = 1:nq
                for j = 1:nx
                    psi[j,1,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[j],obj.settings.yMid[1],n-1);
                    psi[j,end,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[j],obj.settings.yMid[end],n-1);
                end
                for j = 1:ny
                    psi[1,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[1],obj.settings.yMid[j],n-1);
                    psi[end,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[end],obj.settings.yMid[j],n-1);
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi./density,flux);

        psi .= psi .- dE*flux;
        
        psiNew .= psi ./ (1.0+dE*sigmaS[1]);

        # stream collided particles
        uTilde = u .- dE * (obj.L2x*rho0Inv*u*obj.pn.Ax + obj.L2y*rho0Inv*u*obj.pn.Az + obj.L1x*rho0Inv*u*obj.AbsAx' + obj.L1y*rho0Inv*u*obj.AbsAz');
        uTilde = uTilde .- dE * (obj.L2x*rho1InvXi*u*obj.pn.Ax + obj.L2y*rho1InvXi*u*obj.pn.Az + obj.L1x*rho1InvXi*u*obj.AbsAx' + obj.L1y*rho1InvXi*u*obj.AbsAz');  
        #uTilde[obj.boundaryIdx,:] .= 0.0;

        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                idx = vectorIndex(ny,i,j);
                uNew[idx,:] = (Id .+ dE*D)\(uTilde[idx,:] .+ dE*Diagonal(Dvec)*obj.MReduced*psiNew[i,j,:]);
                uOUnc[idx] = psiNew[i,j,:]'*obj.MReduced[1,:];
            end
        end
        
        # update dose
        obj.dose .+= dE * (uNew[:,1]+uOUnc) * obj.csd.SMid[n-1] ./ densityVec ./( 1 + (n==2||n==nEnergies));


        u .= uNew;
        #u[obj.boundaryIdx,:] .= 0.0;
        psi .= psiNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,psi;

end

function SolveFirstCollisionSource(obj::SolverCSD,densityVec::Array{Float64,1})
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries

    # Set up initial condition and store as matrix
    psi = SetupIC(obj);
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam]
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.MReduced = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));
    density = Vec2Mat(nx,ny,densityVec);
    rhoInv = Diagonal(1.0./densityVec);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE


    u = zeros(nx*ny,N);
    uNew = deepcopy(u)
    flux = zeros(size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(nx*ny);

    psiNew = zeros(size(psi));
    uTilde = zeros(size(u))

    obj.dose .= zeros(size(obj.dose));

    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if obj.settings.problem != "validation" # validation testcase sets beam in initial condition
            for k = 1:nq
                for j = 1:nx
                    psi[j,1,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[j],obj.settings.yMid[1],n-1);
                    psi[j,end,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[j],obj.settings.yMid[end],n-1);
                end
                for j = 1:ny
                    psi[1,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[1],obj.settings.yMid[j],n-1);
                    psi[end,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n],obj.settings.xMid[end],obj.settings.yMid[j],n-1);
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi./density,flux);

        psi .= psi .- dE*flux;
        
        psiNew .= psi ./ (1.0+dE*sigmaS[1]);

        # stream collided particles
        uTilde = u .- dE * (obj.L2x*rhoInv*u*obj.pn.Ax + obj.L2y*rhoInv*u*obj.pn.Az + obj.L1x*rhoInv*u*obj.AbsAx' + obj.L1y*rhoInv*u*obj.AbsAz');
        #uTilde[obj.boundaryIdx,:] .= 0.0;

        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                idx = vectorIndex(ny,i,j);
                uNew[idx,:] = (Id .+ dE*D)\(uTilde[idx,:] .+ dE*Diagonal(Dvec)*obj.MReduced*psiNew[i,j,:]);
                uOUnc[idx] = psiNew[i,j,:]'*obj.MReduced[1,:];
            end
        end
        
        # update dose
        obj.dose .+= dE * (uNew[:,1]+uOUnc) * obj.csd.SMid[n-1] ./ densityVec ./( 1 + (n==2||n==nEnergies));


        u .= uNew;
        #u[obj.boundaryIdx,:] .= 0.0;
        psi .= psiNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,psi;

end

function SolveFirstCollisionSourceTensorDLRAold(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    rho0Inv = Diagonal(s.rho0InvVec);
    rho1Inv = Diagonal(s.rho1InvVec);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;

    u = zeros(nx*ny,N,nxi);

    # obtain tensor representation of initial data
    r = obj.settings.r;
    psiTest = ttm(Ten2Ten(psi),obj.MReduced,2);
    TT = hosvd(psiTest,reqrank=[r,r,r]);
    C = TT.cten; C = zeros(r,r,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    W = TT.fmat[2]; W = FillMatrix(W,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    uNew = deepcopy(u);
    flux = zeros(size(psi));

    uOUnc = zeros(nx*ny,nxi);

    psiNew = zeros(size(psi));
    doseXi = zeros(nxi,nx*ny);

    Ax = Matrix(obj.pn.Ax);
    Az = Matrix(obj.pn.Az);
    L2x = Matrix(obj.L2x*rho0Inv);
    L2y = Matrix(obj.L2y*rho0Inv);
    L1x = Matrix(obj.L1x*rho0Inv);
    L1y = Matrix(obj.L1y*rho0Inv);
    L2x1 = Matrix(obj.L2x*rho1Inv);
    L2y1 = Matrix(obj.L2y*rho1Inv);
    L1x1 = Matrix(obj.L1x*rho1Inv);
    L1y1 = Matrix(obj.L1y*rho1Inv);
    AbsAx = Matrix(obj.AbsAx);
    AbsAz = Matrix(obj.AbsAz);

    xi, w = gausslegendre(nxi);
    Xi = Matrix(Diagonal(xi));

    #loop over energy
    prog = Progress(nEnergies-1,1);
    for n=2:nEnergies

        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psiNew .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);

        ################## K1-step ##################
        QT,ST = qr(tenmat(C,1)'); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,1,[r,r,r]); S = Matrix(ST)';
        K = X*S;
        V = ttm(Q,[W,U],[2,3]);
        #println("test u: ",norm(matten(K*tenmat(V,1),1,[nx*ny,N,nxi]) - u))
        #println("test u: ",norm(matten(K*tenmat(V,1)*tenmat(V,1)',1,[nx*ny,N,nxi])-u))

        VVT = tenmat(V,1)'*tenmat(V,1);
        println("test uVVT: ",norm(matten(tenmat(u,1)*VVT,1,[nx*ny,N,nxi]) - u))

        # stream collided particles
        rhs = -ttm(Q, [L2x*K,Ax*W,U],[1,2,3]) -ttm(Q, [L2y*K,Az*W,U],[1,2,3]) - ttm(Q, [L1x*K,AbsAx*W,U],[1,2,3]) -ttm(Q,[L1y*K,AbsAz*W,U],[1,2,3]);
        rhs .+= -ttm(Q,[L2x1*K,Ax*W,Xi*U],[1,2,3]) - ttm(Q,[L2y1*K,Az*W,Xi*U],[1,2,3]) - ttm(Q,[L1x1*K,AbsAx*W,Xi*U],[1,2,3]) - ttm(Q,[L1y1*K,AbsAz*W,Xi*U],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);

        println("before K1-step: ",norm(Var(u)))
        u .= u + dE*rhs;
        println("variance uVVT after time up: ",norm(Var(matten(tenmat(u,1)*VVT,1,[nx*ny,N,nxi]))))
        println("after K1-step: ",norm(Var(u)))
        println("preservation test: ",norm(ones(nxi).- U*U'*ones(nxi)))
        Ktest = K .+ dE*tenmat(ttm(rhs,[Matrix(W'),Matrix(U')],[2,3]),1)*tenmat(Q,1)';
        K = tenmat(u,1)*tenmat(V,1)' # is this here correct?
        uTest = matten(K*tenmat(V,1),1,[nx*ny,N,nxi]); #projection onto V, i.e. u*VV' does not preserve zero variance. Why?
        println("-> var test: ",norm(Var(uTest)))

        println("K test: ",norm(Ktest-K))

        X,S = qr(K);
        X = Matrix(X); S = Matrix(S);
        X = X[:,1:r];
        println("QR test ",norm(X*S-K))

        u = ttm(Q,[K,W,U],[1,2,3]);

        println("K1-step: ",norm(Var(u)))

        rhs = ttm(Q, [L2x*K,Ax*W,U],[1,2,3]) +ttm(Q, [L2y*K,Az*W,U],[1,2,3]) - ttm(Q, [L1x*K,AbsAx*W,U],[1,2,3]) -ttm(Q,[L1y*K,AbsAz*W,U],[1,2,3]);
        rhs .+= ttm(Q,[L2x1*K,Ax*W,Xi*U],[1,2,3]) + ttm(Q,[L2y1*K,Az*W,Xi*U],[1,2,3]) - ttm(Q,[L1x1*K,AbsAx*W,Xi*U],[1,2,3]) - ttm(Q,[L1y1*K,AbsAz*W,Xi*U],[1,2,3]);  
        rhs .-= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);

        u .= u + dE*rhs;
        S = ttm(tenmat(u,1)*tenmat(V,1)',Matrix(X'),1);
        C = matten(S*Matrix(QT)',1,[r,r,r]);

        u = ttm(C,[X,W,U],[1,2,3]);

        println("S1-step: ",norm(Var(u)))

        #u = ttm(Q,[K,W,U],[1,2,3]);

        #u = ttm(u,[W*W',U*U'],[2,3])
        #u = ttm(u,U*U',3)
        #println(norm(uOld-u))
        
        ################## K2-step ##################
        QT,ST = qr(Matrix(tenmat(C,2)')); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,2,[r,r,r]);
        K = W*S;
        V = ttm(Q,[X,U],[1,3]);

        # stream collided particles
        rhs = -ttm(Q, [L2x*X,Ax*K,U],[1,2,3]) -ttm(Q, [L2y*X,Az*K,U],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*K,U],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*K,U],[1,2,3]);
        rhs .+= -ttm(Q,[L2x1*X,Ax*K,Xi*U],[1,2,3]) - ttm(Q,[L2y1*X,Az*K,Xi*U],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*K,Xi*U],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*K,Xi*U],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        u .= u + dE*rhs;
        K = tenmat(u,2)*tenmat(V,2)'

        W,S = qr(K);
        W = Matrix(W); S = Matrix(S);
        W = W[:,1:r];

        u = ttm(Q,[X,K,U],[1,2,3]);

        println("K2-step: ",norm(Var(u)))

        rhs = ttm(Q, [L2x*X,Ax*K,U],[1,2,3]) +ttm(Q, [L2y*X,Az*K,U],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*K,U],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*K,U],[1,2,3]);
        rhs .+= ttm(Q,[L2x1*X,Ax*K,Xi*U],[1,2,3]) + ttm(Q,[L2y1*X,Az*K,Xi*U],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*K,Xi*U],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*K,Xi*U],[1,2,3]);  
        rhs .-= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);

        u .= u + dE*rhs;
        S = ttm(tenmat(u,2)*tenmat(V,2)',Matrix(W'),1); # multiply from front, since 2nd dimension is collected at front

        C = matten(S*Matrix(QT)',2,[r,r,r]);

        u = ttm(C,[X,W,U],[1,2,3]);

        println("S2-step: ",norm(Var(u)))

        ################## C-step ##################
        rhs = -ttm(C, [L2x*X,Ax*W,U],[1,2,3]) -ttm(C, [L2y*X,Az*W,U],[1,2,3]) - ttm(C, [L1x*X,AbsAx*W,U],[1,2,3]) -ttm(C,[L1y*X,AbsAz*W,U],[1,2,3]);
        rhs .+= -ttm(C,[L2x1*X,Ax*W,Xi*U],[1,2,3]) - ttm(C,[L2y1*X,Az*W,Xi*U],[1,2,3]) - ttm(C,[L1x1*X,AbsAx*W,Xi*U],[1,2,3]) - ttm(C,[L1y1*X,AbsAz*W,Xi*U],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        
        C = C .+ dE*ttm(rhs,[Matrix(X'),Matrix(W'),Matrix(U')],[1,2,3])

        println("C-step: ",norm(Var(u)))

        # stream collided particles
        #rhs = -ttm(Q, [L2x*X,Ax*W*S,U],[1,2,3]) -ttm(Q, [L2y*X,Az*W*S,U],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*W*S,U],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*W*S,U],[1,2,3]);
        #rhs .+= -ttm(Q,[L2x1*X,Ax*W*S,Xi*U],[1,2,3]) - ttm(Q,[L2y1*X,Az*W*S,Xi*U],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*W*S,Xi*U],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*W*S,Xi*U],[1,2,3]);  
        #rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        #u .= u + dE*rhs;
        #=
        ################## K3-step ##################
        QT,ST = qr(tenmat(C,3)'); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,3,[r,r,r]);
        K = U*S;

        # stream collided particles
        rhs = -ttm(Q, [L2x*X,Ax*W,U*S],[1,2,3]) -ttm(Q, [L2y*X,Az*W,U*S],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*W,U*S],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*W,U*S],[1,2,3]);
        rhs .+= -ttm(Q,[L2x1*X,Ax*W,Xi*U*S],[1,2,3]) - ttm(Q,[L2y1*X,Az*W,Xi*U*S],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*W,Xi*U*S],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*W,Xi*U*S],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        #u .= u + dE*rhs;
    =#
        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                for k = 1:nxi
                    idx = vectorIndex(ny,i,j);
                    uNew[idx,:,k] = (Id .+ dE*D)\(u[idx,:,k]);
                    uOUnc[idx,k] = psiNew[i,j,:,k]'*obj.MReduced[1,:];
                end
            end
        end

        # update dose
        Phi = uNew[:,1,:]
        for l = 1:nxi
            doseXi[l,:] .+= dE * (Phi[:,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*(s.rho0InvVec.+s.rho1InvVec.*xi[l])./( 1 + (n==2||n==nEnergies));
        end

        u .= uNew;
        psi .= psiNew;

        println("scattering: ",norm(Var(u)))


        #decompose
        TT = hosvd(u,reqrank=[r,r,r]);
        C = TT.cten; C = FillTensor(C,r);
        X = TT.fmat[1]; X = FillMatrix(X,r);
        W = TT.fmat[2]; W = FillMatrix(W,r);
        U = TT.fmat[3]; U = FillMatrix(U,r);

        u = ttm(C,[X,W,U],[1,2,3]);
        println("decompose: ",norm(Var(u)))

        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        obj.dose .+= w[l]*doseXi[l,:]*0.5;
    end

    VarDose = zeros(size(obj.dose));

    # compute dose variance
    for l = 1:nxi
        VarDose .+= 0.5*w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,VarDose,psi;

end

function SolveFirstCollisionSourceTensorDLRA(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    rho0Inv = Diagonal(s.rho0InvVec);
    rho1Inv = Diagonal(s.rho1InvVec);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;

    u = zeros(nx*ny,N,nxi);

    # obtain tensor representation of initial data
    r = obj.settings.r;
    psiTest = ttm(Ten2Ten(psi),obj.MReduced,2);
    TT = hosvd(psiTest,reqrank=[r,r,r]);
    C = TT.cten; C = zeros(r,r,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    W = TT.fmat[2]; W = FillMatrix(W,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    uNew = deepcopy(u);
    flux = zeros(size(psi));

    uOUnc = zeros(nx*ny,nxi);

    psiNew = zeros(size(psi));
    doseXi = zeros(nxi,nx*ny);

    Ax = Matrix(obj.pn.Ax);
    Az = Matrix(obj.pn.Az);
    L2x = Matrix(obj.L2x*rho0Inv);
    L2y = Matrix(obj.L2y*rho0Inv);
    L1x = Matrix(obj.L1x*rho0Inv);
    L1y = Matrix(obj.L1y*rho0Inv);
    L2x1 = Matrix(obj.L2x*rho1Inv);
    L2y1 = Matrix(obj.L2y*rho1Inv);
    L1x1 = Matrix(obj.L1x*rho1Inv);
    L1y1 = Matrix(obj.L1y*rho1Inv);
    AbsAx = Matrix(obj.AbsAx);
    AbsAz = Matrix(obj.AbsAz);

    xi, w = gausslegendre(nxi);
    Xi = Matrix(Diagonal(xi));

    #loop over energy
    prog = Progress(nEnergies-1,1);
    for n=2:nEnergies

        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psiNew .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);

        ################## K1-step ##################
        QT,ST = qr(tenmat(C,1)'); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,1,[r,r,r]); S = Matrix(ST)';
        K = X*S;
        V = ttm(Q,[W,U],[2,3]);

        # stream collided particles
        rhs = -ttm(Q, [L2x*K,Ax*W,U],[1,2,3]) -ttm(Q, [L2y*K,Az*W,U],[1,2,3]) - ttm(Q, [L1x*K,AbsAx*W,U],[1,2,3]) -ttm(Q,[L1y*K,AbsAz*W,U],[1,2,3]);
        rhs .+= -ttm(Q,[L2x1*K,Ax*W,Xi*U],[1,2,3]) - ttm(Q,[L2y1*K,Az*W,Xi*U],[1,2,3]) - ttm(Q,[L1x1*K,AbsAx*W,Xi*U],[1,2,3]) - ttm(Q,[L1y1*K,AbsAz*W,Xi*U],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        u .= u + dE*rhs;
        K = tenmat(u,1)*tenmat(V,1)'

        X,S = qr(K);
        X = Matrix(X); S = Matrix(S);
        X = X[:,1:r];

        rhs = ttm(Q, [L2x*K,Ax*W,U],[1,2,3]) +ttm(Q, [L2y*K,Az*W,U],[1,2,3]) - ttm(Q, [L1x*K,AbsAx*W,U],[1,2,3]) -ttm(Q,[L1y*K,AbsAz*W,U],[1,2,3]);
        rhs .+= ttm(Q,[L2x1*K,Ax*W,Xi*U],[1,2,3]) + ttm(Q,[L2y1*K,Az*W,Xi*U],[1,2,3]) - ttm(Q,[L1x1*K,AbsAx*W,Xi*U],[1,2,3]) - ttm(Q,[L1y1*K,AbsAz*W,Xi*U],[1,2,3]);  
        rhs .-= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);

        u .= u + dE*rhs;
        S = ttm(tenmat(u,1)*tenmat(V,1)',Matrix(X'),1);
        C = matten(S*Matrix(QT)',1,[r,r,r]);


        u = ttm(C,[X,W,U],[1,2,3]);

        #u = ttm(u,[W*W',U*U'],[2,3])
        #u = ttm(u,U*U',3)
        #println(norm(uOld-u))
        
        ################## K2-step ##################
        QT,ST = qr(Matrix(tenmat(C,2)')); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,2,[r,r,r]);
        K = W*S;
        V = ttm(Q,[X,U],[1,3]);

        # stream collided particles
        rhs = -ttm(Q, [L2x*X,Ax*K,U],[1,2,3]) -ttm(Q, [L2y*X,Az*K,U],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*K,U],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*K,U],[1,2,3]);
        rhs .+= -ttm(Q,[L2x1*X,Ax*K,Xi*U],[1,2,3]) - ttm(Q,[L2y1*X,Az*K,Xi*U],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*K,Xi*U],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*K,Xi*U],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        u .= u + dE*rhs;
        K = tenmat(u,2)*tenmat(V,2)'

        W,S = qr(K);
        W = Matrix(W); S = Matrix(S);
        W = W[:,1:r];

        rhs = ttm(Q, [L2x*X,Ax*K,U],[1,2,3]) +ttm(Q, [L2y*X,Az*K,U],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*K,U],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*K,U],[1,2,3]);
        rhs .+= ttm(Q,[L2x1*X,Ax*K,Xi*U],[1,2,3]) + ttm(Q,[L2y1*X,Az*K,Xi*U],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*K,Xi*U],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*K,Xi*U],[1,2,3]);  
        rhs .-= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);

        u .= u + dE*rhs;
        S = ttm(tenmat(u,2)*tenmat(V,2)',Matrix(W'),1); # multiply from front, since 2nd dimension is collected at front

        C = matten(S*Matrix(QT)',2,[r,r,r]);

        u = ttm(C,[X,W,U],[1,2,3]);

        ################## C-step ##################
        rhs = -ttm(C, [L2x*X,Ax*W,U],[1,2,3]) -ttm(C, [L2y*X,Az*W,U],[1,2,3]) - ttm(C, [L1x*X,AbsAx*W,U],[1,2,3]) -ttm(C,[L1y*X,AbsAz*W,U],[1,2,3]);
        rhs .+= -ttm(C,[L2x1*X,Ax*W,Xi*U],[1,2,3]) - ttm(C,[L2y1*X,Az*W,Xi*U],[1,2,3]) - ttm(C,[L1x1*X,AbsAx*W,Xi*U],[1,2,3]) - ttm(C,[L1y1*X,AbsAz*W,Xi*U],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        u .= u + dE*rhs;
        C = ttm(u,[Matrix(X'),Matrix(W'),Matrix(U')],[1,2,3])

        u = ttm(C,[X,W,U],[1,2,3]);

        # stream collided particles
        #rhs = -ttm(Q, [L2x*X,Ax*W*S,U],[1,2,3]) -ttm(Q, [L2y*X,Az*W*S,U],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*W*S,U],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*W*S,U],[1,2,3]);
        #rhs .+= -ttm(Q,[L2x1*X,Ax*W*S,Xi*U],[1,2,3]) - ttm(Q,[L2y1*X,Az*W*S,Xi*U],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*W*S,Xi*U],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*W*S,Xi*U],[1,2,3]);  
        #rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        #u .= u + dE*rhs;
        #=
        ################## K3-step ##################
        QT,ST = qr(tenmat(C,3)'); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,3,[r,r,r]);
        K = U*S;

        # stream collided particles
        rhs = -ttm(Q, [L2x*X,Ax*W,U*S],[1,2,3]) -ttm(Q, [L2y*X,Az*W,U*S],[1,2,3]) - ttm(Q, [L1x*X,AbsAx*W,U*S],[1,2,3]) -ttm(Q,[L1y*X,AbsAz*W,U*S],[1,2,3]);
        rhs .+= -ttm(Q,[L2x1*X,Ax*W,Xi*U*S],[1,2,3]) - ttm(Q,[L2y1*X,Az*W,Xi*U*S],[1,2,3]) - ttm(Q,[L1x1*X,AbsAx*W,Xi*U*S],[1,2,3]) - ttm(Q,[L1y1*X,AbsAz*W,Xi*U*S],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        #u .= u + dE*rhs;
    =#
        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                for k = 1:nxi
                    idx = vectorIndex(ny,i,j);
                    uNew[idx,:,k] = (Id .+ dE*D)\(u[idx,:,k]);
                    uOUnc[idx,k] = psiNew[i,j,:,k]'*obj.MReduced[1,:];
                end
            end
        end

        # update dose
        Phi = uNew[:,1,:]
        for l = 1:nxi
            doseXi[l,:] .+= dE * (Phi[:,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*(s.rho0InvVec.+s.rho1InvVec.*xi[l])./( 1 + (n==2||n==nEnergies));
        end

        u .= uNew;
        psi .= psiNew;


        #decompose
        TT = hosvd(u,reqrank=[r,r,r]);
        C = TT.cten; C = FillTensor(C,r);
        X = TT.fmat[1]; X = FillMatrix(X,r);
        W = TT.fmat[2]; W = FillMatrix(W,r);
        U = TT.fmat[3]; U = FillMatrix(U,r);

        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        obj.dose .+= w[l]*doseXi[l,:]*0.5;
    end

    VarDose = zeros(size(obj.dose));

    # compute dose variance
    for l = 1:nxi
        VarDose .+= 0.5*w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,VarDose,psi;

end

function SolveFirstCollisionSourceTensor(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);
    rho1Inv = Diagonal(s.rho1InvVec);

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;


    u = zeros(nx*ny,N,nxi);
    uNew = deepcopy(u);
    flux = zeros(size(psi));

    uOUnc = zeros(nx*ny,nxi);

    psiNew = zeros(size(psi));
    doseXi = zeros(nxi,nx*ny);

    Ax = Matrix(obj.pn.Ax);
    Az = Matrix(obj.pn.Az);
    L2x = Matrix(obj.L2x*rho0Inv);
    L2y = Matrix(obj.L2y*rho0Inv);
    L1x = Matrix(obj.L1x*rho0Inv);
    L1y = Matrix(obj.L1y*rho0Inv);
    L2x1 = Matrix(obj.L2x*rho1Inv);
    L2y1 = Matrix(obj.L2y*rho1Inv);
    L1x1 = Matrix(obj.L1x*rho1Inv);
    L1y1 = Matrix(obj.L1y*rho1Inv);
    AbsAx = Matrix(obj.AbsAx);
    AbsAz = Matrix(obj.AbsAz);

    xi, w = gausslegendre(nxi);
    w = w*0.5;
    if obj.settings.problem == "timeCT"
        xi = collect(range(0,1,nxi));
        w = 1.0/nxi*ones(size(xi))
    end
    Xi = Matrix(Diagonal(xi));

    #loop over energy
    prog = Progress(nEnergies-1,1);
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psiNew .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);

        # stream collided particles
        rhs = -ttm(u, [L2x,Ax],[1,2]) -ttm(u, [L2y,Az],[1,2]) - ttm(u, [L1x,AbsAx],[1,2]) -ttm(u,[L1y,AbsAz],[1,2]);
        rhs .+= -ttm(u,[L2x1,Ax,Xi],[1,2,3]) - ttm(u,[L2y1,Az,Xi],[1,2,3]) - ttm(u,[L1x1,AbsAx,Xi],[1,2,3]) - ttm(u,[L1y1,AbsAz,Xi],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        u .= u + dE*rhs;
        u[obj.boundaryIdx,:,:] .= 0.0;

        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                for k = 1:nxi
                    idx = vectorIndex(ny,i,j);
                    uNew[idx,:,k] = (Id .+ dE*D)\(u[idx,:,k]);
                    uOUnc[idx,k] = psiNew[i,j,:,k]'*obj.MReduced[1,:];
                end
            end
        end

        u[obj.boundaryIdx,:,:] .= 0.0

        # update dose
        Phi = uNew[:,1,:];
        for l = 1:nxi
            doseXi[l,:] .+= dE * (Phi[:,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*(s.rho0InvVec.+s.rho1InvVec.*xi[l])./( 1 + (n==2||n==nEnergies));
        end

        u .= uNew;
        psi .= psiNew;
        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        #obj.dose .+= w[l]*doseXi[l,:]*0.5;
        obj.dose .+= w[l]*doseXi[l,:];
    end

    VarDose = zeros(size(obj.dose));

    # compute dose variance
    for l = 1:nxi
        #VarDose .+= 0.5*w[l]*(doseXi[l,:] .- obj.dose).^2;
        VarDose .+= w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,VarDose,psi;

end

function SolveFirstCollisionSourceUINaive(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);
    rho1InvXi = Diagonal(s.rho1InvVec).*xi;

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));
    density = (s.rho0Inv .+ s.rho1Inv.*xi).^(-1);
    densityVec = (s.rho0InvVec .+ s.rho1InvVec.*xi).^(-1);

    rho0Inv = Diagonal(s.rho0InvVec);
    rho1Inv = Diagonal(s.rho1InvVec);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;


    u = zeros(nx*ny,N,nxi);
    uNew = deepcopy(u);
    flux = zeros(size(psi));

    # obtain tensor representation of initial data
    r = obj.settings.r;
    psiTest = ttm(Ten2Ten(psi),obj.MReduced,2);
    TT = hosvd(psiTest,reqrank=[r,r,r]);
    C = TT.cten; C = zeros(r,r,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    W = TT.fmat[2]; W = FillMatrix(W,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    uOUnc = zeros(nx*ny,nxi);

    psiNew = zeros(size(psi));
    doseXi = zeros(nxi,nx*ny);

    Ax = Matrix(obj.pn.Ax);
    Az = Matrix(obj.pn.Az);
    L2x = Matrix(obj.L2x*rho0Inv);
    L2y = Matrix(obj.L2y*rho0Inv);
    L1x = Matrix(obj.L1x*rho0Inv);
    L1y = Matrix(obj.L1y*rho0Inv);
    L2x1 = Matrix(obj.L2x*rho1Inv);
    L2y1 = Matrix(obj.L2y*rho1Inv);
    L1x1 = Matrix(obj.L1x*rho1Inv);
    L1y1 = Matrix(obj.L1y*rho1Inv);
    AbsAx = Matrix(obj.AbsAx);
    AbsAz = Matrix(obj.AbsAz);

    xi, w = gausslegendre(nxi);
    Xi = Matrix(Diagonal(xi));

    #loop over energy
    prog = Progress(nEnergies-1,1);
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psiNew .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);

        ################## K1-step ##################
        QT,ST = qr(tenmat(C,1)'); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,1,[r,r,r]); S = Matrix(ST)';
        K = X*S;
        V = ttm(Q,[W,U],[2,3]);
        VVT = tenmat(V,1)'*tenmat(V,1);

        # stream collided particles
        rhs = -ttm(u, [L2x,Ax],[1,2]) -ttm(u, [L2y,Az],[1,2]) - ttm(u, [L1x,AbsAx],[1,2]) -ttm(u,[L1y,AbsAz],[1,2]);
        rhs .+= -ttm(u,[L2x1,Ax,Xi],[1,2,3]) - ttm(u,[L2y1,Az,Xi],[1,2,3]) - ttm(u,[L1x1,AbsAx,Xi],[1,2,3]) - ttm(u,[L1y1,AbsAz,Xi],[1,2,3]);  
        rhs .+= ttm(Ten2Ten(psiNew),Diagonal(Dvec)*obj.MReduced,2);
        u .= u + dE*rhs;
        K = tenmat(u,1)*tenmat(V,1)'

        XNew,S = qr(K);
        XNew = Matrix(XNew); S = Matrix(S);
        XNew = XNew[:,1:r];
        MX = XNew'*X;

        ################## K2-step ##################
        QT,ST = qr(Matrix(tenmat(C,2)')); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,2,[r,r,r]);
        K = W*S;
        V = ttm(Q,[X,U],[1,3]);

        K = tenmat(u,2)*tenmat(V,2)'

        WNew,S = qr(K);
        WNew = Matrix(WNew); S = Matrix(S);
        WNew = WNew[:,1:r];
        MW = WNew'*W;

        ################## K3-step ##################
        QT,ST = qr(Matrix(tenmat(C,3)')); # decompose core tensor
        S = Matrix(ST)';
        Q = (Matrix(QT)');
        Q = matten(Q,3,[r,r,r]);
        K = W*S;
        V = ttm(Q,[X,W],[1,2]);

        K = tenmat(u,3)*tenmat(V,3)'

        UNew,S = qr(K);
        UNew = Matrix(UNew); S = Matrix(S);
        UNew = UNew[:,1:r];
        MU = UNew'*U;

        ################## C-step ##################

        X .= XNew;
        W .= WNew;
        U .= UNew;

        UXiU = U'*Xi*U;

        XL2xX = X'*obj.L2x*rho0Inv*X
        XL2yX = X'*obj.L2y*rho0Inv*X
        XL1xX = X'*obj.L1x*rho0Inv*X
        XL1yX = X'*obj.L1y*rho0Inv*X

        XL2xX1 = X'*obj.L2x*rho1Inv*X
        XL2yX1 = X'*obj.L2y*rho1Inv*X
        XL1xX1 = X'*obj.L1x*rho1Inv*X
        XL1yX1 = X'*obj.L1y*rho1Inv*X

        WAzW = W'*obj.pn.Az*W;
        WAbsAzW = W'*obj.AbsAz*W;
        WAbsAxW = W'*obj.AbsAx*W;
        WAxW = W'*obj.pn.Ax*W;

        C = ttm(C,[MX,MW,MU],[1,2,3]);

        rhsC = - ttm(C,[XL2xX,WAxW],[1,2]) .- ttm(C,[XL2yX,WAzW],[1,2]) .- ttm(C,[XL1xX,WAbsAxW],[1,2]) .- ttm(C,[XL1yX,WAbsAzW],[1,2]);
        rhsC .+= - ttm(C,[XL2xX1,WAxW,UXiU],[1,2,3]) .- ttm(C,[XL2yX1,WAzW,UXiU],[1,2,3]) .- ttm(C,[XL1xX1,WAbsAxW,UXiU],[1,2,3]) .- ttm(C,[XL1yX1,WAbsAzW,UXiU],[1,2,3]);
        rhsC .+= ttm(Ten2Ten(psiNew),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        C = C .+ dE*rhsC;

        u = ttm(C,[X,W,U],[1,2,3]);

        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                for k = 1:nxi
                    idx = vectorIndex(ny,i,j);
                    uNew[idx,:,k] = (Id .+ dE*D)\(u[idx,:,k]);
                    uOUnc[idx,k] = psiNew[i,j,:,k]'*obj.MReduced[1,:];
                end
            end
        end

        # update dose
        Phi = uNew[:,1,:]
        for l = 1:nxi
            doseXi[l,:] .+= dE * (Phi[:,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*(s.rho0InvVec.+s.rho1InvVec.*xi[l])./( 1 + (n==2||n==nEnergies));
        end

        u .= uNew;
        psi .= psiNew;

        #decompose
        TT = hosvd(u,reqrank=[r,r,r]);
        C = TT.cten; C = FillTensor(C,r);
        X = TT.fmat[1]; X = FillMatrix(X,r);
        W = TT.fmat[2]; W = FillMatrix(W,r);
        U = TT.fmat[3]; U = FillMatrix(U,r);
        
        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        obj.dose .+= w[l]*doseXi[l,:]*0.5;
    end

    VarDose = zeros(size(obj.dose));

    # compute dose variance
    for l = 1:nxi
        VarDose .+= 0.5*w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,VarDose,psi;

end

function SolveFirstCollisionSourceUIOld(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    rho0Inv = Diagonal(s.rho0InvVec);
    rho1Inv = Diagonal(s.rho1InvVec);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;


    u = zeros(nx*ny,N,nxi);
    flux = zeros(size(psi));

    # obtain tensor representation of initial data
    r = obj.settings.r;
    psiTest = ttm(Ten2Ten(psi),obj.MReduced,2);
    TT = hosvd(psiTest,reqrank=[r,r,r]);
    C = TT.cten; C = zeros(r,r,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    W = TT.fmat[2]; W = FillMatrix(W,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    uOUnc = zeros(nx*ny,nxi);

    doseXi = zeros(nxi,nx*ny);

    WAxW = zeros(r,r);
    WAzW = zeros(r,r);
    WAbsAxW = zeros(r,r);
    WAbsAzW = zeros(r,r);

    XL2xX = zeros(r,r);
    XL2yX = zeros(r,r);
    XL1xX = zeros(r,r);
    XL1yX = zeros(r,r);
    XL2xX1 = zeros(r,r);
    XL2yX1 = zeros(r,r);
    XL1xX1 = zeros(r,r);
    XL1yX1 = zeros(r,r);

    xi, w = gausslegendre(nxi);
    Xi = Matrix(Diagonal(xi));

    #loop over energy
    prog = Progress(nEnergies-1,1);
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psi .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);

        ################## K1-step ##################
        QT,ST = qr(tenmat(C,1)'); # decompose core tensor
        Q = matten(Matrix(QT)',1,[r,r,r]); S = Matrix(ST)';
        K = X*S;
        K[obj.boundaryIdx,:] .= 0.0;

        WAzW .= W'*obj.pn.Az*W;
        WAbsAzW .= W'*obj.AbsAz*W;
        WAbsAxW .= W'*obj.AbsAx*W;
        WAxW .= W'*obj.pn.Ax*W;
        UXiU = U'*Xi*U;

        rhsK = - ttm(Q,[obj.L2x*rho0Inv*K,WAxW],[1,2]) .- ttm(Q,[obj.L2y*rho0Inv*K,WAzW],[1,2]) .- ttm(Q,[obj.L1x*rho0Inv*K,WAbsAxW],[1,2]) .- ttm(Q,[obj.L1y*rho0Inv*K,WAbsAzW],[1,2]);
        rhsK .+= - ttm(Q,[obj.L2x*rho1Inv*K,WAxW,UXiU],[1,2,3]) .- ttm(Q,[obj.L2y*rho1Inv*K,WAzW,UXiU],[1,2,3]) .- ttm(Q,[obj.L1x*rho1Inv*K,WAbsAxW,UXiU],[1,2,3]) .- ttm(Q,[obj.L1y*rho1Inv*K,WAbsAzW,UXiU],[1,2,3]);
        rhsK .+= ttm(Ten2Ten(psi),[W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[2,3]) # in-scattering from uncollided particles

        K = K .+ dE*tenmat(rhsK,1)*tenmat(Q,1)';

        K[obj.boundaryIdx,:] .= 0.0;

        XNew,S = qr(K);
        XNew = Matrix(XNew); S = Matrix(S);
        XNew = XNew[:,1:r];
        MX = XNew'*X;

        ################## K2-step ##################
        QT,ST = qr(tenmat(C,2)'); # decompose core tensor
        Q = matten(Matrix(QT)',2,[r,r,r]); S = Matrix(ST)';
        K = W*S;

        XL2xX .= X'*obj.L2x*rho0Inv*X
        XL2yX .= X'*obj.L2y*rho0Inv*X
        XL1xX .= X'*obj.L1x*rho0Inv*X
        XL1yX .= X'*obj.L1y*rho0Inv*X

        XL2xX1 .= X'*obj.L2x*rho1Inv*X
        XL2yX1 .= X'*obj.L2y*rho1Inv*X
        XL1xX1 .= X'*obj.L1x*rho1Inv*X
        XL1yX1 .= X'*obj.L1y*rho1Inv*X

        rhsK = - ttm(Q,[XL2xX,obj.pn.Ax*K],[1,2]) .- ttm(Q,[XL2yX,obj.pn.Az*K],[1,2]) .- ttm(Q,[XL1xX,obj.AbsAx*K],[1,2]) .- ttm(Q,[XL1yX,obj.AbsAz*K],[1,2]);
        rhsK .+= - ttm(Q,[XL2xX1,obj.pn.Ax*K,UXiU],[1,2,3]) .- ttm(Q,[XL2yX1,obj.pn.Az*K,UXiU],[1,2,3]) .- ttm(Q,[XL1xX1,obj.AbsAx*K,UXiU],[1,2,3]) .- ttm(Q,[XL1yX1,obj.AbsAz*K,UXiU],[1,2,3]);
        rhsK .+= ttm(Ten2Ten(psi),[Matrix(X'),Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        K = K .+ dE*tenmat(rhsK,2)*tenmat(Q,2)';

        WNew,S = qr(K);
        WNew = Matrix(WNew); S = Matrix(S);
        WNew = WNew[:,1:r];
        MW = WNew'*W;

        ################## K3-step ##################
        QT,ST = qr(tenmat(C,3)'); # decompose core tensor
        Q = matten(Matrix(QT)',3,[r,r,r]); S = Matrix(ST)';
        K = U*S;

        WAzW .= W'*obj.pn.Az*W # Az  = Az^T
        WAbsAzW .= W'*obj.AbsAz*W
        WAbsAxW .= W'*obj.AbsAx*W
        WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T

        rhsK = - ttm(Q,[XL2xX,WAxW,K],[1,2,3]) .- ttm(Q,[XL2yX,WAzW,K],[1,2,3]) .- ttm(Q,[XL1xX,WAbsAxW,K],[1,2,3]) .- ttm(Q,[XL1yX,WAbsAzW,K],[1,2,3]);
        rhsK .+= - ttm(Q,[XL2xX1,WAxW,Xi*K],[1,2,3]) .- ttm(Q,[XL2yX1,WAzW,Xi*K],[1,2,3]) .- ttm(Q,[XL1xX1,WAbsAxW,Xi*K],[1,2,3]) .- ttm(Q,[XL1yX1,WAbsAzW,Xi*K],[1,2,3]);
        rhsK .+= ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced],[1,2]) # in-scattering from uncollided particles
        K = K .+ dE*tenmat(rhsK,3)*tenmat(Q,3)';

        UNew,S = qr(K);
        UNew = Matrix(UNew); S = Matrix(S);
        UNew = UNew[:,1:r];
        MU = UNew'*U;

        ################## C-step ##################

        X .= XNew;
        W .= WNew;
        U .= UNew;

        UXiU = U'*Xi*U;

        XL2xX .= X'*obj.L2x*rho0Inv*X
        XL2yX .= X'*obj.L2y*rho0Inv*X
        XL1xX .= X'*obj.L1x*rho0Inv*X
        XL1yX .= X'*obj.L1y*rho0Inv*X

        XL2xX1 .= X'*obj.L2x*rho1Inv*X
        XL2yX1 .= X'*obj.L2y*rho1Inv*X
        XL1xX1 .= X'*obj.L1x*rho1Inv*X
        XL1yX1 .= X'*obj.L1y*rho1Inv*X

        WAzW .= W'*obj.pn.Az*W;
        WAbsAzW .= W'*obj.AbsAz*W;
        WAbsAxW .= W'*obj.AbsAx*W;
        WAxW .= W'*obj.pn.Ax*W;

        C = ttm(C,[MX,MW,MU],[1,2,3]);

        rhsC = - ttm(C,[XL2xX,WAxW],[1,2]) .- ttm(C,[XL2yX,WAzW],[1,2]) .- ttm(C,[XL1xX,WAbsAxW],[1,2]) .- ttm(C,[XL1yX,WAbsAzW],[1,2]);
        rhsC .+= - ttm(C,[XL2xX1,WAxW,UXiU],[1,2,3]) .- ttm(C,[XL2yX1,WAzW,UXiU],[1,2,3]) .- ttm(C,[XL1xX1,WAbsAxW,UXiU],[1,2,3]) .- ttm(C,[XL1yX1,WAbsAzW,UXiU],[1,2,3]);
        rhsC .+= ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        C = C .+ dE*rhsC;

        ############## Out Scattering ##############
        QT,ST = qr(tenmat(C,2)'); # decompose core tensor such that Mat_2(C) = S*Qmat = QT*ST
        Q = matten(Matrix(QT)',2,[r,r,r]); S = Matrix(ST)';
        L = W*S;

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i] 
        end

        W,S = qr(L);
        W = Matrix(W); S = Matrix(S);
        W = W[:,1:r];

        C = matten(S*Matrix(QT)',2,[r,r,r]);

        # update dose
        Phi = ttm(C,[X,W[1:1,:],U],[1,2,3])
        for l = 1:nxi
            doseXi[l,:] .+= dE * (Phi[:,1,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*(s.rho0InvVec.+s.rho1InvVec.*xi[l])./( 1 + (n==2||n==nEnergies));
        end
        
        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        obj.dose .+= w[l]*doseXi[l,:]*0.5;
    end

    VarDose = zeros(size(obj.dose));

    # compute dose variance
    for l = 1:nxi
        VarDose .+= 0.5*w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,VarDose,psi;

end

function SolveFirstCollisionSourceUI(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    #xi, w = gausslegendre(nxi);
    xi = collect(range(0,1,nxi));
    w = 1.0/nxi*ones(size(xi))

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "timeCT" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            for l = 1:nxi
                psiBeam[k] += PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,xi[l],1)
            end
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;


    u = zeros(nx*ny,N,nxi);
    flux = zeros(size(psi));

    # obtain tensor representation of initial data
    r = obj.settings.r;
    if s.problem == "validation"
        psiTest = ttm(Ten2Ten(psi),obj.MReduced,2);
    else
        psiTest = zeros(size(psi));
        n = 2;
        for l = 1:nxi
            for k = 1:nq
                for j = 1:nx
                    psiTest[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                    psiTest[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                end
                for j = 1:ny
                    psiTest[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                    psiTest[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                end
            end
        end
        psiTest = ttm(Ten2Ten(psiTest),obj.MReduced,2);
    end
    TT = hosvd(psiTest,reqrank=[r,r,r]);
    C = TT.cten; C = zeros(r,r,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    W = TT.fmat[2]; W = FillMatrix(W,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    uOUnc = zeros(nx*ny,nxi);

    doseXi = zeros(nxi,nx*ny);

    WAxW = zeros(r,r);
    WAzW = zeros(r,r);
    WAbsAxW = zeros(r,r);
    WAbsAzW = zeros(r,r);

    XL2xX = zeros(r,r);
    XL2yX = zeros(r,r);
    XL1xX = zeros(r,r);
    XL1yX = zeros(r,r);
    XL2xX1 = zeros(r,r);
    XL2yX1 = zeros(r,r);
    XL1xX1 = zeros(r,r);
    XL1yX1 = zeros(r,r);

    rXi = length(s.rhoInv)
    rhoInv = s.rhoInvX*Diagonal(s.rhoInv)*s.rhoInvXi';


    #loop over energy
    prog = Progress(nEnergies-1,1);
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psi .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);

        ################## K1-step ##################
        QT,ST = qr(tenmat(C,1)'); # decompose core tensor
        Q = matten(Matrix(QT)',1,[r,r,r]); S = Matrix(ST)';
        K = X*S;
        K[obj.boundaryIdx,:] .= 0.0;

        WAzW .= W'*obj.pn.Az*W;
        WAbsAzW .= W'*obj.AbsAz*W;
        WAbsAxW .= W'*obj.AbsAx*W;
        WAxW .= W'*obj.pn.Ax*W;
        

        rhsK = zeros(nx*ny,r,r);
        for k = 1:rXi
            UXiU = U'*(s.rhoInvXi[:,k].*U);
            rhsK .+= s.rhoInv[k]*(- ttm(Q,[obj.L2x*Diagonal(s.rhoInvX[:,k])*K,WAxW,UXiU],[1,2,3]) .- ttm(Q,[obj.L2y*Diagonal(s.rhoInvX[:,k])*K,WAzW,UXiU],[1,2,3]) .- ttm(Q,[obj.L1x*Diagonal(s.rhoInvX[:,k])*K,WAbsAxW,UXiU],[1,2,3]) .- ttm(Q,[obj.L1y*Diagonal(s.rhoInvX[:,k])*K,WAbsAzW,UXiU],[1,2,3]));
        end
        rhsK .+= ttm(Ten2Ten(psi),[W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[2,3]) # in-scattering from uncollided particles

        K = K .+ dE*tenmat(rhsK,1)*tenmat(Q,1)';

        K[obj.boundaryIdx,:] .= 0.0;

        XNew,S = qr(K);
        XNew = Matrix(XNew); S = Matrix(S);
        XNew = XNew[:,1:r];
        MX = XNew'*X;

        ################## K2-step ##################
        QT,ST = qr(tenmat(C,2)'); # decompose core tensor
        Q = matten(Matrix(QT)',2,[r,r,r]); S = Matrix(ST)';
        K = W*S;

        rhsK = zeros(r,N,r);

        for k = 1:rXi
            XL2xX .= X'*obj.L2x*Diagonal(s.rhoInvX[:,k])*X
            XL2yX .= X'*obj.L2y*Diagonal(s.rhoInvX[:,k])*X
            XL1xX .= X'*obj.L1x*Diagonal(s.rhoInvX[:,k])*X
            XL1yX .= X'*obj.L1y*Diagonal(s.rhoInvX[:,k])*X

            UXiU = U'*(s.rhoInvXi[:,k].*U);
            
            rhsK .+= s.rhoInv[k] *(- ttm(Q,[XL2xX,obj.pn.Ax*K,UXiU],[1,2,3]) .- ttm(Q,[XL2yX,obj.pn.Az*K,UXiU],[1,2,3]) .- ttm(Q,[XL1xX,obj.AbsAx*K,UXiU],[1,2,3]) .- ttm(Q,[XL1yX,obj.AbsAz*K,UXiU],[1,2,3]));
        end
        rhsK .+= ttm(Ten2Ten(psi),[Matrix(X'),Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        K = K .+ dE*tenmat(rhsK,2)*tenmat(Q,2)';

        WNew,S = qr(K);
        WNew = Matrix(WNew); S = Matrix(S);
        WNew = WNew[:,1:r];
        MW = WNew'*W;

        ################## K3-step ##################
        QT,ST = qr(tenmat(C,3)'); # decompose core tensor
        Q = matten(Matrix(QT)',3,[r,r,r]); S = Matrix(ST)';
        K = U*S;

        WAzW .= W'*obj.pn.Az*W # Az  = Az^T
        WAbsAzW .= W'*obj.AbsAz*W
        WAbsAxW .= W'*obj.AbsAx*W
        WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T

        rhsK = zeros(r,r,nxi);
        for k = 1:rXi
            XL2xX .= X'*obj.L2x*Diagonal(s.rhoInvX[:,k])*X
            XL2yX .= X'*obj.L2y*Diagonal(s.rhoInvX[:,k])*X
            XL1xX .= X'*obj.L1x*Diagonal(s.rhoInvX[:,k])*X
            XL1yX .= X'*obj.L1y*Diagonal(s.rhoInvX[:,k])*X

            rhsK .+= s.rhoInv[k] *(- ttm(Q,[XL2xX,WAxW,s.rhoInvXi[:,k].*K],[1,2,3]) .- ttm(Q,[XL2yX,WAzW,s.rhoInvXi[:,k].*K],[1,2,3]) .- ttm(Q,[XL1xX,WAbsAxW,s.rhoInvXi[:,k].*K],[1,2,3]) .- ttm(Q,[XL1yX,WAbsAzW,s.rhoInvXi[:,k].*K],[1,2,3]));
        end
        rhsK .+= ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced],[1,2]) # in-scattering from uncollided particles
        K = K .+ dE*tenmat(rhsK,3)*tenmat(Q,3)';

        UNew,S = qr(K);
        UNew = Matrix(UNew); S = Matrix(S);
        UNew = UNew[:,1:r];
        MU = UNew'*U;

        ################## C-step ##################

        X .= XNew;
        W .= WNew;
        U .= UNew;

        WAzW .= W'*obj.pn.Az*W;
        WAbsAzW .= W'*obj.AbsAz*W;
        WAbsAxW .= W'*obj.AbsAx*W;
        WAxW .= W'*obj.pn.Ax*W;

        C = ttm(C,[MX,MW,MU],[1,2,3]);

        rhsC = zeros(size(C));
        for k = 1:rXi
            XL2xX .= X'*obj.L2x*Diagonal(s.rhoInvX[:,k])*X
            XL2yX .= X'*obj.L2y*Diagonal(s.rhoInvX[:,k])*X
            XL1xX .= X'*obj.L1x*Diagonal(s.rhoInvX[:,k])*X
            XL1yX .= X'*obj.L1y*Diagonal(s.rhoInvX[:,k])*X

            UXiU = U'*(s.rhoInvXi[:,k].*U);

            rhsC .+= s.rhoInv[k] *(- ttm(C,[XL2xX,WAxW,UXiU],[1,2,3]) .- ttm(C,[XL2yX,WAzW,UXiU],[1,2,3]) .- ttm(C,[XL1xX,WAbsAxW,UXiU],[1,2,3]) .- ttm(C,[XL1yX,WAbsAzW,UXiU],[1,2,3]));
        end
        rhsC .+= ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        C = C .+ dE*rhsC;

        ############## Out Scattering ##############
        QT,ST = qr(tenmat(C,2)'); # decompose core tensor such that Mat_2(C) = S*Qmat = QT*ST
        Q = matten(Matrix(QT)',2,[r,r,r]); S = Matrix(ST)';
        L = W*S;

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i] 
        end

        W,S = qr(L);
        W = Matrix(W); S = Matrix(S);
        W = W[:,1:r];

        C = matten(S*Matrix(QT)',2,[r,r,r]);

        # update dose
        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                for l = 1:nxi
                    idx = vectorIndex(ny,i,j);
                    uOUnc[idx,l] = psi[i,j,:,l]'*obj.MReduced[1,:];
                end
            end
        end
        Phi = ttm(C,[X,W[1:1,:],U],[1,2,3])
        for l = 1:nxi
            doseXi[l,:] .+= dE * (Phi[:,1,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*rhoInv[:,l]./( 1 + (n==2||n==nEnergies));
        end
        
        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        obj.dose .+= w[l]*doseXi[l,:];
    end

    # compute dose variance
    VarDose = zeros(size(obj.dose));
    for l = 1:nxi
        VarDose .+= w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,VarDose,psi;

end

function SolveFirstCollisionSourceDLR(obj::SolverCSD)
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    nxi = obj.settings.Nxi;

    xi, w = gausslegendre(nxi);
    Xi = Diagonal(xi);

    # Set up initial condition and store as matrix
    psi = zeros(nx,ny,nq,nxi);
    for k = 1:nxi
        psi[:,:,:,k] .= SetupIC(obj);
    end
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" || obj.settings.problem == "lung2" # determine relevant directions in IC
        println(size(psi))
        idxFullBeam = findall(psi .> floorPsiAll)
        println(size(idxFullBeam));
        println(maximum(psi))
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "lung2" || obj.settings.problem == "timeCT"# determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    end
    psi = psi[:,:,idxBeam,:];
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent");
    nq = length(idxBeam);

    # define density matrix
    # store density uncertainties
    rho0Inv = Diagonal(s.rho0InvVec);
    rho1Inv = Diagonal(s.rho1InvVec);
    Id = Diagonal(ones(N));

    # obtain tensor representation of initial data
    psiTest = ttm(Ten2Ten(psi),obj.MReduced,2);
    TT = hosvd(psiTest,reqrank=[r,r,r]);
    C = TT.cten; C = zeros(r,r,r);
    X = TT.fmat[1]; X = FillMatrix(X,r);
    W = TT.fmat[2]; W = FillMatrix(W,r);
    U = TT.fmat[3]; U = FillMatrix(U,r);

    WAxW = zeros(r,r);
    WAzW = zeros(r,r);
    WAbsAxW = zeros(r,r);
    WAbsAzW = zeros(r,r);

    XL2xX = zeros(r,r);
    XL2yX = zeros(r,r);
    XL1xX = zeros(r,r);
    XL1yX = zeros(r,r);
    XL2xX1 = zeros(r,r);
    XL2yX1 = zeros(r,r);
    XL1xX1 = zeros(r,r);
    XL1yX1 = zeros(r,r);

    # impose boundary condition
    #X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;

    flux = zeros(size(psi));
    doseXi = zeros(nxi,nx*ny);

    uOUnc = zeros(nx*ny,s.Nxi);
    
    #loop over energy
    prog = Progress(nEnergies-1,1)
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if s.problem != "validation" # validation testcase sets beam in initial condition
            for l = 1:nxi
                for k = 1:nq
                    for j = 1:nx
                        psi[j,1,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[1],xi[l],n-1);
                        psi[j,end,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[j],s.yMid[end],xi[l],n-1);
                    end
                    for j = 1:ny
                        psi[1,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[1],s.yMid[j],xi[l],n-1);
                        psi[end,j,k,l] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],s.xMid[end],s.yMid[j],xi[l],n-1);
                    end
                end
            end
        end

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        #psiBC = psi[obj.boundaryIdx];

        psi .= (psi .- dE*flux) ./ (1.0+dE*sigmaS[1]);
        #psi[obj.boundaryIdx] .= psiBC; # no scattering in boundary cells
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        ################## K1-step ##################
        QT,ST = qr(tenmat(C,1)'); # decompose core tensor
        Q = matten(Matrix(QT)',1,[r,r,r]); S = Matrix(ST)';
        K = X*S;
        K[obj.boundaryIdx,:] .= 0.0;

        WAzW .= W'*obj.pn.Az*W;
        WAbsAzW .= W'*obj.AbsAz*W;
        WAbsAxW .= W'*obj.AbsAx*W;
        WAxW .= W'*obj.pn.Ax*W;
        UXiU = U'*Xi*U;

        rhsK = - ttm(Q,[obj.L2x*rho0Inv*K,WAxW],[1,2]) .- ttm(Q,[obj.L2y*rho0Inv*K,WAzW],[1,2]) .- ttm(Q,[obj.L1x*rho0Inv*K,WAbsAxW],[1,2]) .- ttm(Q,[obj.L1y*rho0Inv*K,WAbsAzW],[1,2]);
        rhsK .+= - ttm(Q,[obj.L2x*rho1Inv*K,WAxW,UXiU],[1,2,3]) .- ttm(Q,[obj.L2y*rho1Inv*K,WAzW,UXiU],[1,2,3]) .- ttm(Q,[obj.L1x*rho1Inv*K,WAbsAxW,UXiU],[1,2,3]) .- ttm(Q,[obj.L1y*rho1Inv*K,WAbsAzW,UXiU],[1,2,3]);
        rhsK .+= ttm(Ten2Ten(psi),[W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[2,3]) # in-scattering from uncollided particles

        K = K .+ dE*tenmat(rhsK,1)*tenmat(Q,1)';

        K[obj.boundaryIdx,:] .= 0.0;

        X,S = qr(K);
        X = Matrix(X); S = Matrix(S);
        X = X[:,1:r];

        rhsS = + ttm(Q,[X'*obj.L2x*rho0Inv*K,WAxW],[1,2]) .+ ttm(Q,[X'*obj.L2y*rho0Inv*K,WAzW],[1,2]) .- ttm(Q,[X'*obj.L1x*rho0Inv*K,WAbsAxW],[1,2]) .- ttm(Q,[X'*obj.L1y*rho0Inv*K,WAbsAzW],[1,2]);
        rhsS .+= + ttm(Q,[X'*obj.L2x*rho1Inv*K,WAxW,UXiU],[1,2,3]) .+ ttm(Q,[X'*obj.L2y*rho1Inv*K,WAzW,UXiU],[1,2,3]) .- ttm(Q,[X'*obj.L1x*rho1Inv*K,WAbsAxW,UXiU],[1,2,3]) .- ttm(Q,[X'*obj.L1y*rho1Inv*K,WAbsAzW,UXiU],[1,2,3]);
        rhsS .+= -ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        S = S .+ dE* tenmat(rhsS,1)*tenmat(Q,1)'

        C = matten(S*Matrix(QT)',1,[r,r,r]);

        #QT,ST = qr(tenmat(C,3)'); # decompose core tensor
        #Q = matten(Matrix(QT)',3,[r,r,r]); S = Matrix(ST)';
        #K = U*S;
        #println("UQ: ",norm(ttm(Q,[X,W,K],[1,2,3])))
        ################## K2-step ##################
        QT,ST = qr(tenmat(C,2)'); # decompose core tensor
        Q = matten(Matrix(QT)',2,[r,r,r]); S = Matrix(ST)';
        K = W*S;

        XL2xX .= X'*obj.L2x*rho0Inv*X
        XL2yX .= X'*obj.L2y*rho0Inv*X
        XL1xX .= X'*obj.L1x*rho0Inv*X
        XL1yX .= X'*obj.L1y*rho0Inv*X

        XL2xX1 .= X'*obj.L2x*rho1Inv*X
        XL2yX1 .= X'*obj.L2y*rho1Inv*X
        XL1xX1 .= X'*obj.L1x*rho1Inv*X
        XL1yX1 .= X'*obj.L1y*rho1Inv*X

        rhsK = - ttm(Q,[XL2xX,obj.pn.Ax*K],[1,2]) .- ttm(Q,[XL2yX,obj.pn.Az*K],[1,2]) .- ttm(Q,[XL1xX,obj.AbsAx*K],[1,2]) .- ttm(Q,[XL1yX,obj.AbsAz*K],[1,2]);
        rhsK .+= - ttm(Q,[XL2xX1,obj.pn.Ax*K,UXiU],[1,2,3]) .- ttm(Q,[XL2yX1,obj.pn.Az*K,UXiU],[1,2,3]) .- ttm(Q,[XL1xX1,obj.AbsAx*K,UXiU],[1,2,3]) .- ttm(Q,[XL1yX1,obj.AbsAz*K,UXiU],[1,2,3]);
        rhsK .+= ttm(Ten2Ten(psi),[Matrix(X'),Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        K = K .+ dE*tenmat(rhsK,2)*tenmat(Q,2)';

        W,S = qr(K);
        W = Matrix(W); S = Matrix(S);
        W = W[:,1:r];

        rhsS = ttm(Q,[XL2xX,W'*obj.pn.Ax*K],[1,2]) .+ ttm(Q,[XL2yX,W'*obj.pn.Az*K],[1,2]) .- ttm(Q,[XL1xX,W'*obj.AbsAx*K],[1,2]) .- ttm(Q,[XL1yX,W'*obj.AbsAz*K],[1,2]);
        rhsS .+= ttm(Q,[XL2xX1,W'*obj.pn.Ax*K,UXiU],[1,2,3]) .+ ttm(Q,[XL2yX1,W'*obj.pn.Az*K,UXiU],[1,2,3]) .- ttm(Q,[XL1xX1,W'*obj.AbsAx*K,UXiU],[1,2,3]) .- ttm(Q,[XL1yX1,W'*obj.AbsAz*K,UXiU],[1,2,3]);
        rhsS .+= -ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        S = S .+ dE* tenmat(rhsS,2)*tenmat(Q,2)'

        C = matten(S*Matrix(QT)',2,[r,r,r]);

        ################## K3-step ##################
        QT,ST = qr(tenmat(C,3)'); # decompose core tensor
        Q = matten(Matrix(QT)',3,[r,r,r]); S = Matrix(ST)';
        K = U*S;

        WAzW .= W'*obj.pn.Az*W # Az  = Az^T
        WAbsAzW .= W'*obj.AbsAz*W
        WAbsAxW .= W'*obj.AbsAx*W
        WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T

        rhsK = - ttm(Q,[XL2xX1,WAxW,Xi*K],[1,2,3]) .- ttm(Q,[XL2yX1,WAzW,Xi*K],[1,2,3]) .- ttm(Q,[XL1xX1,WAbsAxW,Xi*K],[1,2,3]) .- ttm(Q,[XL1yX1,WAbsAzW,Xi*K],[1,2,3]);
        rhsK .+= ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced],[1,2]) # in-scattering from uncollided particles
        K = K .+ dE*tenmat(rhsK,3)*tenmat(Q,3)';

        U,S = qr(K);
        U = Matrix(U); S = Matrix(S);
        U = U[:,1:r];

        rhsS = ttm(Q,[XL2xX1,WAxW,U'*Xi*K],[1,2,3]) .+ ttm(Q,[XL2yX1,WAzW,U'*Xi*K],[1,2,3]) .- ttm(Q,[XL1xX1,WAbsAxW,U'*Xi*K],[1,2,3]) .- ttm(Q,[XL1yX1,WAbsAzW,U'*Xi*K],[1,2,3]);
        rhsS .+= -ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        S = S .+ dE* tenmat(rhsS,3)*tenmat(Q,3)';

        C = matten(S*Matrix(QT)',3,[r,r,r]);

        ################## C-step ##################

        UXiU = U'*Xi*U;
        
        rhsC = - ttm(C,[XL2xX,WAxW],[1,2]) .- ttm(C,[XL2yX,WAzW],[1,2]) .- ttm(C,[XL1xX,WAbsAxW],[1,2]) .- ttm(C,[XL1yX,WAbsAzW],[1,2]);
        rhsC .+= - ttm(C,[XL2xX1,WAxW,UXiU],[1,2,3]) .- ttm(C,[XL2yX1,WAzW,UXiU],[1,2,3]) .- ttm(C,[XL1xX1,WAbsAxW,UXiU],[1,2,3]) .- ttm(C,[XL1yX1,WAbsAzW,UXiU],[1,2,3]);
        rhsC .+= ttm(Ten2Ten(psi),[Matrix(X'),W'*Diagonal(Dvec)*obj.MReduced,Matrix(U')],[1,2,3]) # in-scattering from uncollided particles
        C = C .+ dE*rhsC;

        ############## Out Scattering ##############
        QT,ST = qr(tenmat(C,2)'); # decompose core tensor such that Mat_3(C) = S*Qmat = QT*ST
        Q = matten(Matrix(QT)',2,[r,r,r]); S = Matrix(ST)';
        L = W*S;

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i] # double check if Q is not needed
        end

        W,S = qr(L);
        W = Matrix(W); S = Matrix(S);
        W = W[:,1:r];

        C = matten(S*Matrix(QT)',2,[r,r,r]);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                for l = 1:nxi
                    uOUnc[idx,l] = psi[i,j,:,l]'*obj.MReduced[1,:];
                end
            end
        end
        Phi = ttm(C,[X,W[1:1,:],U],[1,2,3])
        #println("------------------------")
        for l = 1:nxi
            #println("norm at xi_",l,": ",norm(doseXi[l,:]))
            doseXi[l,:] .+= dE * (Phi[:,1,l] .+ uOUnc[:,l] )* obj.csd.SMid[n-1] .*(s.rho0InvVec.+s.rho1InvVec.*xi[l])./( 1 + (n==2||n==nEnergies));
        end
        
        next!(prog) # update progress bar
    end

    obj.dose .= zeros(size(obj.dose));
    for l = 1:nxi
        obj.dose .+= w[l]*doseXi[l,:]*0.5;
    end

    VarDose = zeros(size(obj.dose));

    # compute dose variance
    for l = 1:nxi
        VarDose .+= 0.5*w[l]*(doseXi[l,:] .- obj.dose).^2;
    end

    TT = hosvd(C,reqrank=[r,r,r]);
    C = TT.cten; C = FillTensor(C,r);
    XU = TT.fmat[1]; X = FillMatrix(X,r);
    WU = TT.fmat[2]; W = FillMatrix(W,r);
    UU = TT.fmat[3]; U = FillMatrix(U,r);
    # return solution and dose
    return X*XU, obj.O*W*WU,U*UU,C,obj.dose,VarDose,psi,doseXi;

end