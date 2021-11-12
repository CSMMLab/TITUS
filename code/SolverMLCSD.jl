__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum

include("CSD.jl")
include("PNSystem.jl")
include("quadratures/Quadrature.jl")

mutable struct SolverMLCSD
    # spatial grid of cell interfaces
    x::Array{Float64};
    y::Array{Float64};
    xGrid::Array{Float64,2}

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # Roe matrix
    AbsAx::Array{Float64,2};
    AbsAz::Array{Float64,2};
    # normalized Legendre Polynomials
    P::Array{Float64,2};
    # quadrature points
    mu::Array{Float64,1};
    w::Array{Float64,1};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # material density
    density::Array{Float64,2};
    densityVec::Array{Float64,1};

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

    rMax::Int;
    L::Int;
    X::Array{Float64,3};
    S::Array{Float64,3};
    W::Array{Float64,3};

    OReduced::Array{Float64,2};
    MReduced::Array{Float64,2};
    qReduced::Array{Float64,2};

    # constructor
    function SolverMLCSD(settings,L=2)
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
        SetupSystemMatrices(pn);

        outRhs = zeros(settings.NCellsX,settings.NCellsY,pn.nTotalEntries);

        # setup Roe matrix
        S = eigvals(pn.Ax)
        V = eigvecs(pn.Ax)
        AbsAx = V*abs.(diagm(S))*inv(V)

        S = eigvals(pn.Az)
        V = eigvecs(pn.Az)
        AbsAz = V*abs.(diagm(S))*inv(V)

        # set density vector
        density = settings.density;

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
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dx/density[i,j]; 
                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx/density[i-1,j];
                end
                if i < nx
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dx/density[i+1,j]; 
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
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dy/density[i,j]; 

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy/density[i,j-1];
                end
                if j < ny
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dy/density[i,j+1]; 
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
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i+1,j);
                indexMinus = vectorIndex(nx,i-1,j);

                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx/density[i-1,j];
                end
                if i < nx
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dx/density[i+1,j];
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
                index = vectorIndex(nx,i,j);
                indexPlus = vectorIndex(nx,i,j+1);
                indexMinus = vectorIndex(nx,i,j-1);

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy/density[i,j-1];
                end
                if j < ny
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dy/density[i,j+1];
                end
            end
        end
        L2y = sparse(II,J,vals,nx*ny,nx*ny);

        # collect boundary indices
        boundaryIdx = zeros(Int,2*nx+2*ny)
        counter = 0;
        for i = 1:nx
            counter +=1;
            j = 1;
            idx = (i-1)*nx + j;
            boundaryIdx[counter] = idx
            counter +=1;
            j = ny;
            idx = (i-1)*nx + j;
            boundaryIdx[counter] = idx
        end

        for j = 1:ny
            counter +=1;
            i = 1;
            idx = (i-1)*nx + j;
            boundaryIdx[counter] = idx
            counter +=1;
            i = nx;
            idx = (i-1)*nx + j;
            boundaryIdx[counter] = idx
        end

        boundaryBeam = zeros(Int,2*nx) # boundary indices uncollided particles for beam
        counter = 0;
        for i = 1:nx
            counter += 1;
            j = 1;
            idx = (i-1)*nx + j;
            boundaryBeam[counter] = idx
            counter += 1;
            j = 2;
            idx = (i-1)*nx + j;
            boundaryBeam[counter] = idx
        end

        # setup spatial grid
        xGrid = zeros(nx*ny,2)
        for i = 1:nx
            for j = 1:ny
                # y part
                index = vectorIndex(nx,i,j);
                xGrid[index,1] = settings.xMid[i];
                xGrid[index,2] = settings.yMid[j];
            end
        end

        # setup quadrature
        qorder = settings.nPN+1; # must be even for standard quadrature
        qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
        Q = Quadrature(qorder,qtype)

        Ωs = Q.pointsxyz
        weights = Q.weights
        #Norder = (qorder+1)*(qorder+1)
        Norder = pn.nTotalEntries
        nq = length(weights);

        #Y = zeros(qorder +1,2*(qorder +1)+1,nq)
        YY = zeros(Norder,nq)
        @polyvar xx yy zz
        counter = 1
        for l=0:settings.nPN
            for m=-l:l
                sphericalh = ylm(l,m,xx,yy,zz)
                for q = 1 : nq
                    #Y[l+1,m+l+1,q] = sphericalh(xx=>Ωs[q,1],yy=>Ωs[q,2],zz=>Ωs[q,3])
                    YY[counter,q] =  sphericalh(xx=>Ωs[q,1],yy=>Ωs[q,2],zz=>Ωs[q,3])
                end
                counter += 1
            end
        end
        normY = zeros(Norder);
        for i = 1:Norder
            for k = 1:nq
                normY[i] = normY[i] + YY[i,k]^2
            end
        end
        normY = sqrt.(normY)
    
        Mat = zeros(nq,nq)
        O = zeros(nq,Norder)
        M = zeros(Norder,nq)
        for i=1:nq
            for j=1:nq
                counter = 1
                for l=0:settings.nPN
                    for m=-l:l
                        O[i,counter] = YY[counter,i] # / normY[counter] 
                        M[counter,j] = YY[counter,j]*weights[j]# / normY[counter]
                        counter += 1
                    end
                end
            end
        end
        #sqrt((2*l+1)/(4*pi).*factorial(l-ma)./factorial(l+ma)).*(-1).^max(m,0).*exp(1i*m*phi).*z(ma+1);
        
        v = O*M*ones(nq)
    
        counter = 1
        for q=1:nq
            O[q,:] /= v[q]
        end

        # allocate solution factors with full rank
        rMax = Int(floor(settings.r/2));
        X = zeros(L,nx*ny,rMax);
        W = zeros(L,N,rMax);
        S = zeros(L,rMax,rMax);

        new(x,y,xGrid,settings,gamma,AbsAx,AbsAz,P,mu,w,csd,pn,density,vec(density'),dose,L1x,L1y,L2x,L2y,boundaryIdx,boundaryBeam,Q,O,M,rMax,L,X,S,W);
    end
end

function SetupIC(obj::SolverMLCSD)
    nq = obj.Q.nquadpoints;
    psi = zeros(obj.settings.NCellsX,obj.settings.NCellsY,nq); 
    for k = 1:nq
        psi[:,:,k] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)
    end
    return psi;
end

function PsiBeam(obj::SolverMLCSD,Omega::Array{Float64,1},E::Float64,x::Float64,y::Float64,n::Int)
    E0 = obj.settings.eMax;
    if obj.settings.problem == "lung"
        sigmaO1Inv = 0.0;
        sigmaO3Inv = 75.0;
        sigmaXInv = 20.0;
        sigmaYInv = 20.0;
        sigmaEInv = 100.0;
    elseif obj.settings.problem == "liver"
        sigmaO1Inv = 10.0;
        sigmaO3Inv = 0.0;
        sigmaXInv = 10.0;
        sigmaYInv = 10.0;
        sigmaEInv = 10.0;
    elseif obj.settings.problem == "LineSource"
        return 0.0;
    end
    return 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2)*exp(-sigmaEInv*(E0-E)^2)*exp(-sigmaXInv*(x-obj.settings.x0)^2)*exp(-sigmaYInv*(y-obj.settings.y0)^2)*obj.csd.S[n]*obj.settings.densityMin;
end

function solveFluxUpwind!(obj::SolverMLCSD, phi::Array{Float64,3}, flux::Array{Float64,3})
    # computes the numerical flux over cell boundaries for each ordinate
    # for faster computation, we split the iteration over quadrature points
    # into four different blocks: North West, Nort East, Sout West, South East
    # this corresponds to the direction the ordinates point to
    idxPosPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].>=0.0))
    idxPosNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].<0.0))
    idxNegPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].>=0.0))
    idxNegNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].<0.0))

    nx = collect(2:(obj.settings.NCellsX-1));
    ny = collect(2:(obj.settings.NCellsY-1));

    # PosPos
    for j=nx,i=ny, q = idxPosPos
        s2 = phi[i,j-1,q]
        s3 = phi[i,j,q]
        northflux = s3
        southflux = s2

        s2 = phi[i-1,j,q]
        s3 = phi[i,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,2]./obj.settings.dy .* (northflux-southflux)
    end
    #PosNeg
    for j=nx,i=ny,q = idxPosNeg
        s2 = phi[i,j,q]
        s3 = phi[i,j+1,q]
        northflux = s3
        southflux = s2

        s2 = phi[i-1,j,q]
        s3 = phi[i,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegPos
    for j=nx,i=ny,q = idxNegPos
        s2 = phi[i,j-1,q]
        s3 = phi[i,j,q]
        northflux = s3
        southflux = s2

        s2 = phi[i,j,q]
        s3 = phi[i+1,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegNeg
    for j=nx,i=ny,q = idxNegNeg
        s2 = phi[i,j,q]
        s3 = phi[i,j+1,q]
        northflux = s3
        southflux = s2

        s2 = phi[i,j,q]
        s3 = phi[i+1,j,q]
        eastflux = s3
        westflux = s2

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux)
    end
end

function UnconventionalIntegratorAdaptive!(obj::SolverMLCSD,Dvec::Array{Float64,1},D,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},psiNew::Array{Float64,3},step::Int,eIndex::Int)
    rmin = 2;
    rMaxTotal = Int(floor(obj.settings.r/2));
    SigmaT = D+Diagonal(Dvec)
    dE = obj.settings.dE;

    X,S,W = UpdateUIStreamingAdaptive(obj,X,S,W);

    r = size(S,1)

    ############## In Scattering ##############
    sigT = SigmaT[1];
    ################## K-step ##################
    X[obj.boundaryIdx,:] .= 0.0;
    K = X*S;
    K .= (K .+dE*Mat2Vec(psiNew)*obj.MReduced'*Diagonal(Dvec)*W)/(1+dE*sigT);
    K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

    XNew,STmp = qr!([K X]);
    XNew = Matrix(XNew)
    XNew = XNew[:,1:2*r];

    MUp = XNew' * X;

    ################## L-step ##################
    L = W*S';
    L .= (L .+dE*(Mat2Vec(psiNew)*obj.MReduced'*Diagonal(Dvec))'*X)/(1+dE*sigT);

    WNew,STmp = qr([L W]);
    WNew = Matrix(WNew)
    WNew = WNew[:,1:2*r];

    NUp = WNew' * W;

    W = WNew;
    X = XNew;

    ################## S-step ##################
    S = MUp*S*(NUp')
    S .= (S .+dE*X'*Mat2Vec(psiNew)*obj.MReduced'*Diagonal(Dvec)*W)/(1+dE*sigT);

    ################## truncate ##################

    # Compute singular values of S1 and decide how to truncate:
    U,D,V = svd(S);
    U = Matrix(U); V = Matrix(V)
    rmax = -1;
    S .= zeros(size(S));

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D)^obj.settings.adaptIndex;
    
    rmax = Int(floor(size(D,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if(tmp<tol)
            rmax = j;
            break;
        end
    end
    
    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rmin);

    for l = 1:rmax
        S[l,l] = D[l];
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    # update solution with new rank
    XNew = XNew*U;
    WNew = WNew*V;

    # impose boundary condition
    XNew[obj.boundaryIdx,:] .= 0.0;

    # update solution with new rank
    obj.S[step,1:rmax,1:rmax] = S[1:rmax,1:rmax];
    obj.X[step,:,1:rmax] = XNew[:,1:rmax];
    obj.W[step,:,1:rmax] = WNew[:,1:rmax];

    return rmax;
end

function UnconventionalIntegratorAdaptive!(obj::SolverMLCSD,Dvec::Array{Float64,1},D,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},XPrev::Array{Float64,2},SPrev::Array{Float64,2},WPrev::Array{Float64,2},step::Int,eIndex::Int)
    rmin = 2;
    rMaxTotal = Int(floor(obj.settings.r/2));
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    SigmaT = D+Diagonal(Dvec)
    dE = obj.settings.dE;
    n = eIndex;
    nEnergies = length(obj.csd.eTrafo);

    N = obj.pn.nTotalEntries
    Id = Diagonal(ones(N));

    X,S,W = UpdateUIStreamingAdaptive(obj,X,S,W);
    r = size(S,1);

    ############## In Scattering ##############
    sigT = SigmaT[1]
    ################## K-step ##################
    X[obj.boundaryIdx,:] .= 0.0;
    K = X*S;
    WPrevDW = WPrev'*Diagonal(Dvec)*W;
    #K .= K .+dE*XPrev*SPrev*WPrevDW;
    K = (K + dE*XPrev*SPrev*WPrev'*Diagonal(Dvec)*W)/(1+dE*sigT)
    #u = u + dE*XPrev*SPrev*WPrev'*Diagonal(Dvec)
    K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

    XNew,STmp = qr!([K X]);
    XNew = Matrix(XNew)
    XNew = XNew[:,1:2*r];

    MUp = XNew' * X;

    ################## L-step ##################
    L = W*S';
    XX = XPrev'*X;
    #L .= L .+dE*Diagonal(Dvec)*WPrev*SPrev'*XX;
    sigT = SigmaT[1]
    L .= (L .+ dE*(XPrev*SPrev*WPrev'*Diagonal(Dvec))'*X)/(1+dE*sigT)

    WNew,STmp = qr([L W]);
    WNew = Matrix(WNew)
    WNew = WNew[:,1:2*r];

    NUp = WNew' * W;
    W = WNew;
    X = XNew;

    ################## S-step ##################
    S = MUp*S*(NUp')
    WPrevDW = WPrev'*Diagonal(Dvec)*W;
    XX = X'*XPrev;

    S .= (S + dE*X'*XPrev*SPrev*WPrev'*Diagonal(Dvec)*W)/(1+dE*sigT)

    ################## truncate ##################

    # Compute singular values of S1 and decide how to truncate:
    U,D,V = svd(S);
    U = Matrix(U); V = Matrix(V)
    rmax = -1;
    S .= zeros(size(S));

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D)^obj.settings.adaptIndex;
    
    rmax = Int(floor(size(D,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if(tmp<tol)
            rmax = j;
            break;
        end
    end
    
    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rmin);

    for l = 1:rmax
        S[l,l] = D[l];
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    # update solution with new rank
    XNew = XNew*U;
    WNew = WNew*V;

    # update solution with new rank
    S = S[1:rmax,1:rmax];
    X = XNew[:,1:rmax];
    W = WNew[:,1:rmax];

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;    

    # update solution with new rank
    obj.S[step,1:rmax,1:rmax] = S[1:rmax,1:rmax];
    obj.X[step,:,1:rmax] = XNew[:,1:rmax];
    obj.W[step,:,1:rmax] = WNew[:,1:rmax];

    return rmax;
end

function UnconventionalIntegratorCollidedAdaptive!(obj::SolverMLCSD,Dvec::Array{Float64,1},D,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},XPrev::Array{Float64,2},SPrev::Array{Float64,2},WPrev::Array{Float64,2},step::Int,eIndex::Int)
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    rmin = 2;
    rMaxTotal = Int(floor(obj.settings.r/2));
    dE = obj.settings.dE;
    N = obj.pn.nTotalEntries
    Id = Diagonal(ones(N));
    nEnergies = length(obj.csd.eTrafo);

    X,S,W = UpdateUIStreamingAdaptive(obj,X,S,W);
    r = size(S,1);

    ############## In Scattering ##############
    SigmaT = D+Diagonal(Dvec)
    sigT = SigmaT[1]
    ################## K-step ##################
    X[obj.boundaryIdx,:] .= 0.0;
    K = X*S;
    WPrevDW = WPrev'*Diagonal(Dvec)*W;
    #K .= K .+dE*XPrev*SPrev*WPrevDW;
    K = K + dE*XPrev*SPrev*WPrev'*Diagonal(Dvec)*W
    #u = u + dE*XPrev*SPrev*WPrev'*Diagonal(Dvec)
    K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

    XNew,STmp = qr!([K X]);
    XNew = Matrix(XNew)
    XNew = XNew[:,1:2*r];

    MUp = XNew' * X;

    ################## L-step ##################
    L = W*S';
    XX = XPrev'*X;
    #L .= L .+dE*Diagonal(Dvec)*WPrev*SPrev'*XX;
    L .= L .+ dE*(XPrev*SPrev*WPrev'*Diagonal(Dvec))'*X

    WNew,STmp = qr([L W]);
    WNew = Matrix(WNew)
    WNew = WNew[:,1:2*r];

    NUp = WNew' * W;
    W = WNew;
    X = XNew;

    ################## S-step ##################
    S = MUp*S*(NUp')
    WPrevDW = WPrev'*Diagonal(Dvec)*W;
    XX = X'*XPrev;

    #S .= S .+dE*XX*SPrev*WPrevDW;
    S = S + dE*X'*XPrev*SPrev*WPrev'*Diagonal(Dvec)*W

    ################## truncate ##################

    # Compute singular values of S1 and decide how to truncate:
    U,DS,V = svd(S);
    U = Matrix(U); V = Matrix(V)
    rmax = -1;
    S .= zeros(size(S));

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(DS);
    
    rmax = Int(floor(size(DS,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(DS[j:2*rmax]).^2);
        if(tmp<tol)
            rmax = j;
            break;
        end
    end
    
    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rmin);

    for l = 1:rmax
        S[l,l] = DS[l];
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    # update solution with new rank
    XNew = XNew*U;
    WNew = WNew*V;

    # update solution with new rank
    S = S[1:rmax,1:rmax];
    X = XNew[:,1:rmax];
    W = WNew[:,1:rmax];

    r = rmax;

    ############## Self-In and Out Scattering ##############
    L = W*S';

    for i = 1:size(L,2)
        L[:,i] = L[:,i]./(1 .+dE*sigT.-dE*Dvec);
    end

    W,S = qr(L);
    W = Matrix(W)
    W = W[:, 1:r];
    S = Matrix(S)
    S = S[1:r, 1:r];

    S .= S';

    # update solution with new rank
    obj.S[step,1:r,1:r] = S;
    obj.X[step,:,1:r] = X;
    obj.W[step,:,1:r] = W;

    return r;
end

function UpdateUIStreamingAdaptive(obj::SolverMLCSD,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},sigmaT::Float64=0.0)
    dE = obj.settings.dE
    r=size(X,2);
    rmin = 2;
    rMaxTotal = Int(floor(obj.settings.r/2));
    #Diagonal(ones(size(Dvec))./(1 .- dE*Dvec))*L[:,i];
    ################## K-step ##################
    K = X*S;

    WAzW = W'*obj.pn.Az'*W
    WAbsAzW = W'*obj.AbsAz'*W
    WAbsAxW = W'*obj.AbsAx'*W
    WAxW = W'*obj.pn.Ax'*W

    K .= (K .- dE*(obj.L2x*K*WAxW + obj.L2y*K*WAzW + obj.L1x*K*WAbsAxW + obj.L1y*K*WAbsAzW))/(1+dE*sigmaT);

    XNew,STmp = qr!([K X]);
    XNew = Matrix(XNew)
    XNew = XNew[:,1:2*r];

    # impose boundary condition
    XNew[obj.boundaryIdx,:] .= 0.0;

    MUp = XNew' * X;
    ################## L-step ##################
    L = W*S';

    XL2xX = X'*obj.L2x*X
    XL2yX = X'*obj.L2y*X
    XL1xX = X'*obj.L1x*X
    XL1yX = X'*obj.L1y*X

    L .= (L .- dE*(obj.pn.Ax*L*XL2xX' + obj.pn.Az*L*XL2yX' + obj.AbsAx*L*XL1xX' + obj.AbsAz*L*XL1yX'))/(1+dE*sigmaT);
            
    WNew,STmp = qr([L W]);
    WNew = Matrix(WNew)
    WNew = WNew[:,1:2*r];

    NUp = WNew' * W;
    W = WNew;
    X = XNew;
    ################## S-step ##################
    S = MUp*S*(NUp')

    XL2xX = X'*obj.L2x*X
    XL2yX = X'*obj.L2y*X
    XL1xX = X'*obj.L1x*X
    XL1yX = X'*obj.L1y*X

    WAzW = W'*obj.pn.Az'*W
    WAbsAzW = W'*obj.AbsAz'*W
    WAbsAxW = W'*obj.AbsAx'*W
    WAxW = W'*obj.pn.Ax'*W

    S .= (S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAzW))/(1+dE*sigmaT);

    ################## truncate ##################

    # Compute singular values of S1 and decide how to truncate:
    U,D,V = svd(S);
    U = Matrix(U); V = Matrix(V)
    rmax = -1;
    S .= zeros(size(S));

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D)^obj.settings.adaptIndex;
    
    rmax = Int(floor(size(D,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if(tmp<tol)
            rmax = j;
            break;
        end
    end
    
    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rmin);

    for l = 1:rmax
        S[l,l] = D[l];
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    # update solution with new rank
    XNew = XNew*U;
    WNew = WNew*V;

    # update solution with new rank
    S = S[1:rmax,1:rmax];
    X = XNew[:,1:rmax];
    W = WNew[:,1:rmax];

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;
    
    return X,S,W;
end

function SolveMCollisionSourceDLR(obj::SolverMLCSD)
    # Get rank
    r=15;
    L = obj.L;
    remap = 1.0;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries

    # Set up initial condition and store as matrix
    psi = SetupIC(obj);
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD"# determine relevant directions in IC
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
    elseif obj.settings.problem == "lung" || obj.settings.problem == "liver" # determine relevant directions in beam
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

    # Low-rank approx of init data:
    X1,S1,W1 = svd(zeros(nx*ny,N));
    
    # rank-r truncation:
    X = zeros(L,nx*ny,r);
    W = zeros(L,N,r);
    for l = 1:L
        X[l,:,:] = X1[:,1:r];
        W[l,:,:] = W1[:,1:r];
    end
    S = zeros(L,r,r);

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(Diagonal(1.0 ./obj.density)))

    flux = zeros(size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(nx*ny);
    
    psiNew = deepcopy(psi);

    rankInTime = zeros(L+1,nEnergies);
    rankInTime[1,1] = energy[1];
    rankInTime[2:end,1] .= r;
    ranks::Array{Int,1} = r*ones(L)

    #loop over energy
    for n=2:nEnergies
        rankInTime[1,n] = energy[n];
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
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

        # stream uncollided particles
        solveFluxUpwind!(obj,psi,flux);

        psi .= psi .- dE*flux;

        psiNew .= psi ./ (1+dE*sigmaS[1]);
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        ranks[1] = UnconventionalIntegratorAdaptive!(obj,Dvec,D,obj.X[1,:,1:ranks[1]],obj.S[1,1:ranks[1],1:ranks[1]],obj.W[1,:,1:ranks[1]],psiNew,1,n)
        obj.X[1,obj.boundaryIdx,:] .= 0.0;
        rankInTime[2,n] = ranks[1];

        for l = 2:(L-1)
            ranks[l] = UnconventionalIntegratorAdaptive!(obj,Dvec,D,obj.X[l,:,1:ranks[l]],obj.S[l,1:ranks[l],1:ranks[l]],obj.W[l,:,1:ranks[l]],obj.X[l-1,:,1:ranks[l-1]],obj.S[l-1,1:ranks[l-1],1:ranks[l-1]],obj.W[l-1,:,1:ranks[l-1]],l,n)
            obj.X[l,obj.boundaryIdx,:] .= 0.0;
            rankInTime[l+1,n] = ranks[l];
        end

        ranks[L] = UnconventionalIntegratorCollidedAdaptive!(obj,Dvec,D,obj.X[L,:,1:ranks[L]],obj.S[L,1:ranks[L],1:ranks[L]],obj.W[L,:,1:ranks[L]],obj.X[L-1,:,1:ranks[L-1]],obj.S[L-1,1:ranks[L-1],1:ranks[L-1]],obj.W[L-1,:,1:ranks[L-1]],L,n)
        obj.X[L,obj.boundaryIdx,:] .= 0.0;
        rankInTime[L+1,n] = ranks[L];

        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*nx + j
                uOUnc[idx] = psiNew[i,j,:]'*obj.MReduced[1,:];
            end
        end
        
        # update dose
        obj.dose .+= dE * uOUnc * obj.csd.SMid[n-1] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));

        for l = 1:L
            obj.dose .+= dE * obj.X[l,:,1:ranks[l]]*obj.S[l,1:ranks[l],1:ranks[l]]*obj.W[l,1,1:ranks[l]] * obj.csd.SMid[n-1] ./ obj.densityVec ./( 1 + (n==2||n==nEnergies));
        end

        # remap strategy
        #WOr = obj.W[L,:,1:ranks[L]]'*obj.OReduced';
        #obj.W[L,:,1:ranks[L]] .= obj.W[L,:,1:ranks[L]] - remap*(WOr*obj.MReduced')';

        #W,S = qr(obj.W[L,:,1:ranks[L]]);
        #W = Matrix(W)
        #obj.W[L,:,1:ranks[L]] = W[:, 1:ranks[L]];
        #S = Matrix(S)
        #S = S[1:ranks[L], 1:ranks[L]];
        #obj.S[L,1:ranks[L],1:ranks[L]] .= obj.S[L,1:ranks[L],1:ranks[L]]*S';

        
        psi .= psiNew;# + remap*Vec2Mat(nx,ny,obj.X[L,:,1:ranks[L]]*obj.S[L,1:ranks[L],1:ranks[L]]*WOr);
        next!(prog) # update progress bar
    end
    U,Sigma,V = svd(obj.S[L,1:ranks[L],1:ranks[L]]);
    # return solution and dose
    return obj.X[L,:,1:ranks[L]]*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*obj.W[L,:,1:ranks[L]]*V,obj.dose,rankInTime,psi;

end

function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
end

function Vec2Mat(nx,ny,v::Array{Float64,1})
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*nx + j]
        end
    end
    return m;
end

function Vec2Mat(nx,ny,v::Array{Float64,2})
    n = size(v,2);
    m = zeros(nx,ny,n);
    for i = 1:nx
        for j = 1:ny
            m[i,j,:] = v[(i-1)*nx + j,:]
        end
    end
    return m;
end

function Mat2Vec(mat)
    nx = size(mat,1)
    ny = size(mat,2)
    m = size(mat,3)
    v = zeros(nx*ny,m);
    for i = 1:nx
        for j = 1:ny
            v[(i-1)*nx + j,:] = mat[i,j,:]
        end
    end
    return v;
end