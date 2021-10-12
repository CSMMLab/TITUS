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

struct SolverCSD
    # spatial grid of cell interfaces
    x::Array{Float64};
    y::Array{Float64};

    # Solver settings
    settings::Settings;

    # preallocate memory for performance
    outRhs::Array{Float64,3};
    
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

    Q::Quadrature
    O::Array{Float64,2};
    M::Array{Float64,2};

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
        
        v = O*M*ones(nq)
    
        counter = 1
        for q=1:nq
            O[q,:] /= v[q]
        end

        new(x,y,settings,outRhs,gamma,AbsAx,AbsAz,P,mu,w,csd,pn,density,vec(density'),dose,L1x,L1y,L2x,L2y,Q,O,M);
    end
end

function SetupIC(obj::SolverCSD)
    nq = obj.Q.nquadpoints;
    psi = zeros(obj.settings.NCellsX,obj.settings.NCellsY,nq);
    
    if obj.settings.problem == "CT" || obj.settings.problem == "2D" || obj.settings.problem == "2DHighD"
        for k = 1:nq
            if obj.Q.pointsxyz[k][1] > 0.5
                psi[:,:,k] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)
            end
        end
    end
    return psi;
end

function PsiLeft(obj::SolverCSD,n::Int,mu::Float64)
    E0 = obj.settings.eMax;
    return 10^5*exp(-200.0*(1.0-mu)^2)*exp(-50*(E0-E)^2)
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

function Slope(u,v,w,dx)
    mMinus = (v-u)/dx;
    mPlus = (w-v)/dx;
    if mPlus*mMinus > 0
        return 2*mMinus*mPlus/(mMinus+mPlus);
    else
        return 0.0;
    end
end

function RhsHighOrder(obj::SolverCSD,u::Array{Float64,3},t::Float64=0.0)   
    #Boundary conditions
    #obj.outRhs[1,:] = u[1,:];
    #obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];
    dx = obj.settings.dx
    dz = obj.settings.dy

    for j=2:obj.settings.NCellsX-1
        for i=2:obj.settings.NCellsY-1
            # Idea: use this formulation to define outRhsX and outRhsY, then apply splitting to get u2 = u + dt*outRhsX(u), uNew = u2 + dt*outRhsY(u2)
            obj.outRhs[j,i,:] = 1/2/dx * obj.pn.Ax * (u[j+1,i,:]/obj.density[j+1,i]-u[j-1,i,:]/obj.density[j-1,i]) - 1/2/dx * obj.AbsAx*( u[j+1,i,:]/obj.density[j+1,i] - 2*u[j,i,:]/obj.density[j,i] + u[j-1,i,:]/obj.density[j-1,i] );
            obj.outRhs[j,i,:] += 1/2/dz * obj.pn.Az * (u[j,i+1,:]/obj.density[j,i+1]-u[j,i-1,:]/obj.density[j,i-1]) - 1/2/dz * obj.AbsAz*( u[j,i+1,:]/obj.density[j,i+1] - 2*u[j,i,:]/obj.density[j,i] + u[j,i-1,:]/obj.density[j,i-1] );
        end
    end
    return obj.outRhs;
end

function RhsMatrix(obj::SolverCSD,u::Array{Float64,3},t::Float64=0.0)   
    #Boundary conditions
    #obj.outRhs[1,:] = u[1,:];
    #obj.outRhs[obj.settings.NCells,:] = u[obj.settings.NCells,:];
    dx = obj.settings.dx
    dz = obj.settings.dy

    for j=2:obj.settings.NCellsX-1
        for i=2:obj.settings.NCellsY-1
            # Idea: use this formulation to define outRhsX and outRhsY, then apply splitting to get u2 = u + dt*outRhsX(u), uNew = u2 + dt*outRhsY(u2)
            obj.outRhs[j,i,:] = 1/2/dx * obj.pn.Ax * (u[j+1,i,:]/obj.density[j+1,i]-u[j-1,i,:]/obj.density[j-1,i]) - 1/2/dx * obj.AbsAx*( u[j+1,i,:]/obj.density[j+1,i] - 2*u[j,i,:]/obj.density[j,i] + u[j-1,i,:]/obj.density[j-1,i] );
            obj.outRhs[j,i,:] += 1/2/dz * obj.pn.Az * (u[j,i+1,:]/obj.density[j,i+1]-u[j,i-1,:]/obj.density[j,i-1]) - 1/2/dz * obj.AbsAz*( u[j,i+1,:]/obj.density[j,i+1] - 2*u[j,i,:]/obj.density[j,i] + u[j,i-1,:]/obj.density[j,i-1] );
        end
    end
    return obj.outRhs;
end

function Rhs(obj::SolverCSD,u::Array{Float64,2},t::Float64=0.0)   
    return obj.L2x*u*obj.pn.Ax' + obj.L2y*u*obj.pn.Az' + obj.L1x*u*obj.AbsAx' + obj.L1y*u*obj.AbsAz';
end

# the first minmod code is the fast version of the second minmod  below that is commented
@inline minmod(x::Float64, y::Float64) = ifelse(x < 0, clamp(y, x, 0.0), clamp(y, 0.0, x))
#@inline function minmod(x::Float64, y::Float64)
#    return sign(x) * max(0.0, min(abs(x),y*sign(x) ) );
#end

@inline function slopefit(left::Float64, center::Float64, right::Float64)
    tmp = minmod(0.5 * (right - left),2.0 * (center - left));
    return minmod(2.0 * (right - center),tmp);
end

function solveFlux!(obj::SolverCSD, phi::Array{Float64,3}, flux::Array{Float64,3})
    # computes the numerical flux over cell boundaries for each ordinate
    # for faster computation, we split the iteration over quadrature points
    # into four different blocks: North West, Nort East, Sout West, South East
    # this corresponds to the direction the ordinates point to
    idxPosPos = findall((obj.Q.pointsxyz[:,1].>=0.0) .&(obj.Q.pointsxyz[:,2].>=0.0))
    idxPosNeg = findall((obj.Q.pointsxyz[:,1].>=0.0) .&(obj.Q.pointsxyz[:,2].<0.0))
    idxNegPos = findall((obj.Q.pointsxyz[:,1].<0.0)  .&(obj.Q.pointsxyz[:,2].>=0.0))
    idxNegNeg = findall((obj.Q.pointsxyz[:,1].<0.0)  .&(obj.Q.pointsxyz[:,2].<0.0))

    nx = collect(3:(obj.settings.NCellsX-2));
    ny = collect(3:(obj.settings.NCellsY-2));

    # PosPos
    for j=nx,i=ny, q = idxPosPos
        s1 = phi[i,j-2,q]
        s2 = phi[i,j-1,q]
        s3 = phi[i,j,q]
        s4 = phi[i,j+1,q]
        northflux = s3+0.5 .*slopefit(s2,s3,s4)
        southflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i-2,j,q]
        s2 = phi[i-1,j,q]
        s3 = phi[i,j,q]
        s4 = phi[i+1,j,q]
        eastflux = s3+0.5 .*slopefit(s2,s3,s4)
        westflux = s2+0.5 .*slopefit(s1,s2,s3)

        flux[i,j,q] = obj.Q.pointsxyz[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.Q.pointsxyz[q,2]./obj.settings.dy .* (northflux-southflux)
    end
    #PosNeg
    for j=nx,i=ny,q = idxPosNeg
        s1 = phi[i,j-1,q]
        s2 = phi[i,j,q]
        s3 = phi[i,j+1,q]
        s4 = phi[i,j+2,q]
        northflux = s3-0.5 .* slopefit(s2,s3,s4)
        southflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i-2,j,q]
        s2 = phi[i-1,j,q]
        s3 = phi[i,j,q]
        s4 = phi[i+1,j,q]
        eastflux = s3+0.5 .*slopefit(s2,s3,s4)
        westflux = s2+0.5 .*slopefit(s1,s2,s3)

        flux[i,j,q] = obj.Q.pointsxyz[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.Q.pointsxyz[q,2] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegPos
    for j=nx,i=ny,q = idxNegPos
        s1 = phi[i,j-2,q]
        s2 = phi[i,j-1,q]
        s3 = phi[i,j,q]
        s4 = phi[i,j+1,q]
        northflux = s3+0.5 .*slopefit(s2,s3,s4)
        southflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i-1,j,q]
        s2 = phi[i,j,q]
        s3 = phi[i+1,j,q]
        s4 = phi[i+2,j,q]
        eastflux = s3-0.5 .*slopefit(s2,s3,s4)
        westflux = s2-0.5 .*slopefit(s1,s2,s3)

        flux[i,j,q] = obj.Q.pointsxyz[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.Q.pointsxyz[q,2] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegNeg
    for j=nx,i=ny,q = idxNegNeg
        s1 = phi[i,j-1,q]
        s2 = phi[i,j,q]
        s3 = phi[i,j+1,q]
        s4 = phi[i,j+2,q]
        northflux = s3-0.5 .* slopefit(s2,s3,s4)
        southflux = s2-0.5 .* slopefit(s1,s2,s3)

        s1 = phi[i-1,j,q]
        s2 = phi[i,j,q]
        s3 = phi[i+1,j,q]
        s4 = phi[i+2,j,q]
        eastflux = s3-0.5 .* slopefit(s2,s3,s4)
        westflux = s2-0.5 .* slopefit(s1,s2,s3)

        flux[i,j,q] = obj.Q.pointsxyz[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.Q.pointsxyz[q,2] ./obj.settings.dy .*(northflux-southflux)
    end
end

function SolveFirstCollisionSource(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    # Set up initial condition and store as matrix
    psi = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    u = zeros(nx*ny,N);
    uNew = deepcopy(u)
    psiNew = deepcopy(psi)
    flux = zeros(size(psi))

    prog = Progress(nEnergies,1)
    scatSN = zeros(size(psi))
    MapOrdinates = obj.O*obj.M

    uOUnc = zeros(nx*ny);

    #loop over energy
    for n=1:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # stream uncollided particles
        solveFlux!(obj,psi,flux);

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

        # set boundary condition
        #u[1,:] .= BCLeft(obj,n);

        # perform time update
        uTilde = u .- dE * Rhs(obj,u); 

        #uTilde[1,:] .= BCLeft(obj,n);
        #uNew = uTilde .- dE*uTilde*D;
        
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*nx + j
                uNew[idx,:] = (Id .+ dE*D)\(uTilde[idx,:] .+ dE*Diagonal(Dvec)*obj.M*psiNew[i,j,:]);
                uOUnc[idx] = psiNew[i,j,:]'*obj.M[1,:];
            end
        end

        #println(maximum(psi)," ",maximum(abs.(uOUnc))," ",maximum(abs.(uNew[:,1])))
        
        # update dose
        obj.dose .+= dE * (uNew[:,1]+uOUnc) * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));
        #if n > 1 && n < nEnergies
        #    obj.dose .+= 0.5 * dE * ( uNew[:,:,1] * S[n] + u[:,:,1] * S[n - 1] ) ./ obj.density;    # update dose with trapezoidal rule
            #obj.dose .+= dE * uNew[:,:,1] * SMinus[n] ./ obj.density;
        #else
        #    obj.dose .+= 0.5*dE * uNew[:,:,1] * S[n] ./ obj.density;
        #end

        u .= uNew;
        psi .= psiNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose;

end

function SolveFirstCollisionSourceDLR(obj::SolverCSD)
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    # Set up initial condition and store as matrix
    psi = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries
    # Set up initial condition and store as matrix
    u = zeros(nx*ny,N);
    for k = 1:N
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*nx + j
                for q = 1:nq
                    u[idx,k] += obj.M[k,q]*psi[i,j,q];
                end
            end
        end
    end

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = zeros(r,r);
    K = zeros(size(X));

    WAxW = zeros(r,r)
    WAzW = zeros(r,r)
    WAbsAxW = zeros(r,r)
    WAbsAzW = zeros(r,r)

    XL2xX = zeros(r,r)
    XL2yX = zeros(r,r)
    XL1xX = zeros(r,r)
    XL1yX = zeros(r,r)

    MUp = zeros(r,r)
    NUp = zeros(r,r)

    XNew = zeros(nx*ny,r)
    STmp = zeros(r,r)

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    
    uNew = deepcopy(u)
    psiNew = deepcopy(psi)
    flux = zeros(size(psi))

    prog = Progress(nEnergies,1)
    scatSN = zeros(size(psi))
    MapOrdinates = obj.O*obj.M

    uOUnc = zeros(nx*ny);

    #loop over energy
    for n=1:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # stream uncollided particles
        solveFlux!(obj,psi,flux);

        #@einsum scatSN[i,j,k] = MapOrdinates[k,l]*psi[i,j,l]*sigmaS[1]

        psi .= psi .- dE*flux;

        #@einsum scatSN[i,j,k] = MapOrdinates[k,l]*psi[i,j,l]*sigmaS[1]
        
        psiNew .= psi ./ (1+dE*sigmaS[1]);
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        ############## Scattering ##############
        L = W*S';

        XTPsi = zeros(nq,r)
        for k = 1:nq
            for l = 1:r
                for i = 1:nx
                    for j = 1:ny
                        idx = (i-1)*nx + j
                        XTPsi[k,l] = XTPsi[k,l] + psiNew[i,j,k]*X[idx,l]
                    end
                end
            end
        end

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\(L[:,i].+dE*Diagonal(Dvec)*obj.M*XTPsi[:,i])
        end

        W,S = qr(L);
        W = Matrix(W)
        W = W[:, 1:r];
        S = Matrix(S)
        S = S[1:r, 1:r];

        S .= S';

        ################## K-step ##################
        K .= X*S;

        WAzW .= W'*obj.pn.Az'*W
        WAbsAzW .= W'*obj.AbsAz'*W
        WAbsAxW .= W'*obj.AbsAx'*W
        WAxW .= W'*obj.pn.Ax'*W

        K .= K .- dE*(obj.L2x*K*WAxW + obj.L2y*K*WAzW + obj.L1x*K*WAbsAxW + obj.L1y*K*WAbsAzW);

        XNew,STmp = qr!(K);
        XNew = Matrix(XNew)
        XNew = XNew[:,1:r];

        MUp .= XNew' * X;
        ################## L-step ##################
        L = W*S';

        XL2xX .= X'*obj.L2x*X
        XL2yX .= X'*obj.L2y*X
        XL1xX .= X'*obj.L1x*X
        XL1yX .= X'*obj.L1y*X

        L .= L .- dE*(obj.pn.Ax*L*XL2xX' + obj.pn.Az*L*XL2yX' + obj.AbsAx*L*XL1xX' + obj.AbsAz*L*XL1yX');
                
        WNew,STmp = qr(L);
        WNew = Matrix(WNew)
        WNew = WNew[:,1:r];

        NUp .= WNew' * W;
        W .= WNew;
        X .= XNew;
        ################## S-step ##################
        S .= MUp*S*(NUp')

        XL2xX .= X'*obj.L2x*X
        XL2yX .= X'*obj.L2y*X
        XL1xX .= X'*obj.L1x*X
        XL1yX .= X'*obj.L1y*X

        WAzW .= W'*obj.pn.Az'*W
        WAbsAzW .= W'*obj.AbsAz'*W
        WAbsAxW .= W'*obj.AbsAx'*W
        WAxW .= W'*obj.pn.Ax'*W

        S .= S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAzW);
       
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*nx + j
                uOUnc[idx] = psiNew[i,j,:]'*obj.M[1,:];
            end
        end

        #println(maximum(psi)," ",maximum(abs.(uOUnc))," ",maximum(abs.(uNew[:,1])))
        
        # update dose
        #obj.dose .+= dE * (uNew[:,1]+uOUnc) * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));
        next!(prog) # update progress bar
        # update dose
        obj.dose .+= dE * (X*S*W[1,:]+uOUnc) * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));

        psi .= psiNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*X*S*W',obj.dose;

end


function SolveFirstCollisionSourceAdaptiveDLR(obj::SolverCSD)
    # Get rank
    r=10;
    rMaxTotal = Int(floor(obj.settings.r/2));

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    # Set up initial condition and store as matrix
    psi = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries
    # Set up initial condition and store as matrix
    u = zeros(nx*ny,N);
    for k = 1:N
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*nx + j
                for q = 1:nq
                    u[idx,k] += obj.M[k,q]*psi[i,j,q];
                end
            end
        end
    end

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = zeros(r,r);
    K = zeros(size(X));

    XNew = zeros(nx*ny,r)
    STmp = zeros(r,r)

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))
    
    uNew = deepcopy(u)
    psiNew = deepcopy(psi)
    flux = zeros(size(psi))

    prog = Progress(nEnergies,1)
    scatSN = zeros(size(psi))
    MapOrdinates = obj.O*obj.M

    rankInTime = zeros(2,nEnergies);

    uOUnc = zeros(nx*ny);

    #loop over energy
    for n=1:nEnergies
        rankInTime[1,n] = energy[n];
        rankInTime[2,n] = r;
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # stream uncollided particles
        solveFlux!(obj,psi,flux);

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

        ############## Scattering ##############
        L = W*S';
        WOld = W;

        XTPsi = zeros(nq,r)
        for k = 1:nq
            for l = 1:r
                for i = 1:nx
                    for j = 1:ny
                        idx = (i-1)*nx + j
                        XTPsi[k,l] = XTPsi[k,l] + psiNew[i,j,k]*X[idx,l]
                    end
                end
            end
        end

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\(L[:,i].+dE*Diagonal(Dvec)*obj.M*XTPsi[:,i])
        end

        W,S = qr(L);
        W = Matrix(W)
        W = W[:, 1:r];
        S = Matrix(S)
        S = S[1:r, 1:r];

        S .= S';

        ################## K-step ##################
        K = X*S;

        WAzW = W'*obj.pn.Az'*W
        WAbsAzW = W'*obj.AbsAz'*W
        WAbsAxW = W'*obj.AbsAx'*W
        WAxW = W'*obj.pn.Ax'*W

        K .= K .- dE*(obj.L2x*K*WAxW + obj.L2y*K*WAzW + obj.L1x*K*WAbsAxW + obj.L1y*K*WAbsAzW);

        XNew,STmp = qr!([K X]);
        XNew = Matrix(XNew)
        XNew = XNew[:,1:2*r];

        MUp = XNew' * X;
        ################## L-step ##################
        L = W*S';

        XL2xX = X'*obj.L2x*X
        XL2yX = X'*obj.L2y*X
        XL1xX = X'*obj.L1x*X
        XL1yX = X'*obj.L1y*X

        L .= L .- dE*(obj.pn.Ax*L*XL2xX' + obj.pn.Az*L*XL2yX' + obj.AbsAx*L*XL1xX' + obj.AbsAz*L*XL1yX');
                
        WNew,STmp = qr([L WOld]);
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

        S .= S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAzW);
       
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*nx + j
                uOUnc[idx] = psiNew[i,j,:]'*obj.M[1,:];
            end
        end
        
        # update dose
        next!(prog) # update progress bar
        # update dose
        obj.dose .+= dE * (X*S*W[1,:]+uOUnc) * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));

        ################## truncate ##################

        # Compute singular values of S1 and decide how to truncate:
        U,D,V = svd(S);
        U = Matrix(U); V = Matrix(V)
        rmax = -1;
        S .= zeros(size(S));

        tmp = 0.0;
        tol = obj.settings.epsAdapt*norm(D);
        
        rmax = Int(floor(size(D,1)/2));
        
        for j=1:2*rmax
            tmp = sqrt(sum(D[j:2*rmax]).^2);
            if(tmp<tol)
                rmax = j;
                break;
            end
        end
        
        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,2);

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

        # update rank
        r = rmax;

        psi .= psiNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*X*S*W',obj.dose,rankInTime;

end


function Solve(obj::SolverCSD)
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    uNew = deepcopy(u)

    prog = Progress(nEnergies,1)

    out = zeros(nx*ny,N);
    vout = RhsMatrix(obj,v)
    for k = 1:N
        out[:,k] = vec(vout[:,:,k]');
    end

    println("error rhs = ",norm(Rhs(obj,u) - out))
    println("norm rhs = ",norm(Rhs(obj,u)))
    println("norm rhs = ",norm(out))

    #loop over energy
    for n=1:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # set boundary condition
        #u[1,:] .= BCLeft(obj,n);

        # perform time update
        uTilde = u .- dE * Rhs(obj,u); 

        # apply filtering
        #lam = 0.0#5e-7
        #for j = 1:(settings.NCellsX-1)
        #    for i = 1:(settings.NCellsY-1)
        #        for k = 1:obj.settings.nPN
        #            uTilde[j,i,k] = uTilde[j,i,k]/(1+lam*k^2*(k-1)^2);
        #        end
        #    end
        #end

        #uTilde[1,:] .= BCLeft(obj,n);
        #uNew = uTilde .- dE*uTilde*D;
        for j = 1:size(uNew,1)
            uNew[j,:] = (Id .+ dE*D)\uTilde[j,:];
        end
        
        # update dose
        obj.dose .+= dE * uNew[:,1] * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));

        u .= uNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose;

end

function SolveNaiveUnconventional(obj::SolverCSD)
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    uNew = deepcopy(u)

    out = zeros(nx*ny,N);
    vout = RhsMatrix(obj,v)
    for k = 1:N
        out[:,k] = vec(vout[:,:,k]');
    end

    println("error rhs = ",norm(Rhs(obj,u) - out))
    println("norm rhs = ",norm(Rhs(obj,u)))
    println("norm rhs = ",norm(out))

    prog = Progress(nEnergies,1)

    for n=1:nEnergies

        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        ################## K-step ##################
        K .= X*S;

        K .= K .- dE*(obj.L2x*u*obj.pn.Ax' + obj.L2y*u*obj.pn.Az' + obj.L1x*u*obj.AbsAx' + obj.L1y*u*obj.AbsAz')*W;

        XNew,STmp = qr(K);
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';

        L .= L .- dE*(X'*(obj.L2x*u*obj.pn.Ax' + obj.L2y*u*obj.pn.Az' + obj.L1x*u*obj.AbsAx' + obj.L1y*u*obj.AbsAz'))';
                
        WNew,STmp = qr(L);
        WNew = WNew[:,1:r];

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        S .= S .- dE.*X'*(obj.L2x*u*obj.pn.Ax' + obj.L2y*u*obj.pn.Az' + obj.L1x*u*obj.AbsAx' + obj.L1y*u*obj.AbsAz')*W;

        #println(maximum(S))

        u .= X*S*W';

        ################## scattering ##################
        #for j = 1:(obj.settings.NCells-1)
        #    u[j,:] = (I + obj.settings.dE *D)\u[j,:];
        #end
        #X,S,W = svd(u);

        for j = 1:size(uNew,1)
            u[j,:] = (Id .+ dE*D)\u[j,:];
        end
        X,S,W = svd(u);

        # rank-r truncation:
        X = X[:,1:r];
        W = W[:,1:r];
        S = Diagonal(S);
        S = S[1:r, 1:r];

        next!(prog) # update progress bar
               
        # update dose
        obj.dose .+= dE * uNew[:,1] * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));
    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*X*S*W',obj.dose;

end

function SolveUnconventional(obj::SolverCSD)
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    # Set up initial condition and store as matrix
    v = SetupIC(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    uNew = deepcopy(u)

    out = zeros(nx*ny,N);
    vout = RhsMatrix(obj,v)
    for k = 1:N
        out[:,k] = vec(vout[:,:,k]');
    end

    WAxW = zeros(r,r)
    WAzW = zeros(r,r)
    WAbsAxW = zeros(r,r)
    WAbsAzW = zeros(r,r)

    XL2xX = zeros(r,r)
    XL2yX = zeros(r,r)
    XL1xX = zeros(r,r)
    XL1yX = zeros(r,r)

    MUp = zeros(r,r)
    NUp = zeros(r,r)

    XNew = zeros(nx*ny,r)
    STmp = zeros(r,r)

    println("error rhs = ",norm(Rhs(obj,u) - out))
    println("norm rhs = ",norm(Rhs(obj,u)))
    println("norm rhs = ",norm(out))

    prog = Progress(nEnergies,1)

    for n=1:nEnergies

        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        ################## K-step ##################
        K .= X*S;

        WAzW .= W'*obj.pn.Az'*W
        WAbsAzW .= W'*obj.AbsAz'*W
        WAbsAxW .= W'*obj.AbsAx'*W
        WAxW .= W'*obj.pn.Ax'*W

        K .= K .- dE*(obj.L2x*K*WAxW + obj.L2y*K*WAzW + obj.L1x*K*WAbsAxW + obj.L1y*K*WAbsAzW);

        XNew,STmp = qr!(K);
        XNew = Matrix(XNew)
        XNew = XNew[:,1:r];

        MUp .= XNew' * X;
        ################## L-step ##################
        L = W*S';

        XL2xX .= X'*obj.L2x*X
        XL2yX .= X'*obj.L2y*X
        XL1xX .= X'*obj.L1x*X
        XL1yX .= X'*obj.L1y*X

        L .= L .- dE*(obj.pn.Ax*L*XL2xX' + obj.pn.Az*L*XL2yX' + obj.AbsAx*L*XL1xX' + obj.AbsAz*L*XL1yX');
                
        WNew,STmp = qr(L);
        WNew = Matrix(WNew)
        WNew = WNew[:,1:r];

        NUp .= WNew' * W;
        W .= WNew;
        X .= XNew;
        ################## S-step ##################
        S .= MUp*S*(NUp')

        XL2xX .= X'*obj.L2x*X
        XL2yX .= X'*obj.L2y*X
        XL1xX .= X'*obj.L1x*X
        XL1yX .= X'*obj.L1y*X

        WAzW .= W'*obj.pn.Az'*W
        WAbsAzW .= W'*obj.AbsAz'*W
        WAbsAxW .= W'*obj.AbsAx'*W
        WAxW .= W'*obj.pn.Ax'*W

        S .= S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAzW);
        ############## Scattering ##############
        L .= W*S';
        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
            LNew = L - dt*LNew*D
        end

        W,S = qr(L);
        W = Matrix(W)
        W = W[:, 1:r];
        S = Matrix(S)
        S = S[1:r, 1:r];

        S .= S';

        next!(prog) # update progress bar
        # update dose
        obj.dose .+= dE * X*S*W[1,:] * obj.csd.SMid[n] ./ obj.densityVec ./( 1 + (n==1||n==nEnergies));

    end

    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*X*S*W',obj.dose;

end

function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
end

function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*nx + j]
        end
    end
    return m;
end