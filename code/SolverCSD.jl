__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions,SphericalHarmonics,TypedPolynomials,GSL
using MultivariatePolynomials
using Einsum
using CUDA
using CUDA.CUSPARSE
using Base.Threads

include("CSD.jl")
include("PNSystem.jl")
include("quadratures/Quadrature.jl")
include("utils.jl")
include("stencils.jl")

mutable struct SolverCSD{T<:AbstractFloat}
    # spatial grid of cell interfaces
    x::Array{T};
    y::Array{T};
    z::Array{T};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{T,1};
    # Roe matrix
    AbsAx::SparseMatrixCSC{T, Int64};
    AbsAy::SparseMatrixCSC{T, Int64};
    AbsAz::SparseMatrixCSC{T, Int64};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # stencil matrices
    stencil::Stencils;

    # material density
    density::Array{T,3};
    densityVec::Array{T,1};

    # dose vector
    dose::Array{T,1};

    boundaryIdx::Array{Int,1}

    Q::Quadrature
    O::Array{T,2};
    M::Array{T,2};

    T::DataType;

    OReduced::Array{T,2};
    MReduced::Array{T,2};
    qReduced::Array{T,2};

    # constructor
    function SolverCSD(settings)
        T = Float32; # define accuracy 
        x = settings.x;
        y = settings.y;
        z = settings.z;

        nx = settings.NCellsX;
        ny = settings.NCellsY;
        nz = settings.NCellsZ;

        # setup flux matrix
        gamma = zeros(T,settings.nPN+1);
        for i = 1:settings.nPN+1
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end

        # construct CSD fields
        csd = CSD(settings,T);

        # construct PN system matrices
        pn = PNSystem(settings,T)
        Ax,Ay,Az = SetupSystemMatrices(pn);
        SetupSystemMatricesSparse(pn);

        # setup Roe matrix
        S = eigvals(Ax)
        V = eigvecs(Ax)
        AbsAx = V*abs.(Diagonal(S))*inv(V)
        idx = findall(abs.(AbsAx) .> 1e-10)
        Ix = first.(Tuple.(idx)); Jx = last.(Tuple.(idx)); vals = AbsAx[idx];
        AbsAx = sparse(Ix,Jx,T.(vals),pn.nTotalEntries,pn.nTotalEntries);

        S = eigvals(Ay)
        V = eigvecs(Ay)
        AbsAy = V*abs.(Diagonal(S))*inv(V)
        idx = findall(abs.(AbsAy) .> 1e-10)
        Iy = first.(Tuple.(idx)); Jy = last.(Tuple.(idx)); valsy = AbsAy[idx];
        AbsAy = sparse(Iy,Jy,T.(valsy),pn.nTotalEntries,pn.nTotalEntries);
        
        S = eigvals(Az)
        V = eigvecs(Az)
        AbsAz = V*abs.(Diagonal(S))*inv(V)
        idx = findall(abs.(AbsAz) .> 1e-10)
        Iz = first.(Tuple.(idx)); Jz = last.(Tuple.(idx)); valsz = AbsAz[idx];
        AbsAz = sparse(Iz,Jz,T.(valsz),pn.nTotalEntries,pn.nTotalEntries);

        # set density vector
        density = T.(settings.density);

        # allocate dose vector
        dose = zeros(T,nx*ny*nz)

        # collect boundary indices
        boundaryIdx = zeros(Int,2*nx*ny+2*ny*nz + 2*nx*nz)
        counter = 0;
        for i = 1:nx
            for k = 1:nz
                counter +=1;
                j = 1;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
                counter +=1;
                j = ny;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
            end
        end

        for i = 1:nx
            for j = 1:ny
                counter +=1;
                k = 1;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
                counter +=1;
                k = nz;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
            end
        end

        for j = 1:ny
            for k = 1:nz
                counter +=1;
                i = 1;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
                counter +=1;
                i = nx;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
            end
        end

        # setup quadrature
        qorder = settings.nPN+22; 
        if iseven(qorder) qorder += 1; end # make quadrature odd to ensure direction (0,1,0) is contained
        qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
        Q = Quadrature(qorder,qtype);

        Norder = pn.nTotalEntries;
        O,M = ComputeTrafoMatrices(Q,Norder,settings.nPN);

        
        stencil = Stencils(settings,T,2);

        densityVec = Ten2Vec(density);

        new{T}(T.(x),T.(y),T.(z),settings,gamma,AbsAx,AbsAy,AbsAz,csd,pn,stencil,density,densityVec,dose,boundaryIdx,Q,T.(O),T.(M),T);
    end
end

function SetupIC(obj::SolverCSD{T},pointsxyz::Matrix{Float64}) where {T<:AbstractFloat}
    nq = size(pointsxyz)[1];
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    psi = zeros(T,obj.settings.NCellsX,obj.settings.NCellsY,obj.settings.NCellsZ,nq);

    if obj.settings.problem == "validation"
        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    for q = 1:nq 
                        psi[i,j,k,q] = PsiBeam(obj,T.(pointsxyz[q,:]),T(0.0),obj.settings.xMid[i],obj.settings.yMid[j],obj.settings.zMid[k],1)
                    end
                end
            end
        end
    else    
        for k = 1:nq
            psi[:,:,:,k] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid,obj.settings.zMid)
        end
    end
    
    return psi;
end

function SetupICMoments(obj::SolverCSD{T}) where {T<:AbstractFloat}
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
                pos_beam = [0.5*14.5,0.5*14.5,0];
                space_beam = normpdf(obj.settings.xMid[i],pos_beam[1],.01).*normpdf(obj.settings.yMid[j],pos_beam[2],.01);
                trafo = obj.csd.S[1]*obj.settings.density[i,j];
                u[i,j,:] = Float64.(obj.pn.M*psi)*space_beam;
            end
        end
    elseif obj.settings.problem == "2D"  || obj.settings.problem == "2DHighLowD"
        for l = 0:obj.settings.nPN
            for k=-l:l
                i = GlobalIndex( l, k )+1;
                u[:,:,i] = IC(obj.settings,obj.settings.xMid,obj.settings.yMid)*obj.csd.StarMAPmoments[i]
            end
        end
    end
    return u;
end

function PsiBeam(obj::SolverCSD{T},Omega::Array{T,1},E::T,x::Float64,y::Float64,z::Float64,n::Int) where {T<:AbstractFloat}
    E0 = obj.settings.eMax;
    if obj.settings.problem == "lung" || obj.settings.problem == "lungOrig"
        sigmaO1Inv = 0.0;
        sigmaO3Inv = 75.0;
        sigmaXInv = 20.0;
        sigmaYInv = 20.0;
        sigmaEInv = 100.0;
    elseif obj.settings.problem == "liver"
        sigmaO1Inv = 75.0;
        sigmaO3Inv = 0.0;
        sigmaXInv = 10.0;
        sigmaYInv = 10.0;
        sigmaEInv = 10.0;
    elseif obj.settings.problem == "validation"
        sigmaO1Inv = 10000.0;
        sigmaO2Inv = 10000.0;
        sigmaO3Inv = 10000.0;
        sigmaEInv = 1000.0;
        pos_beam = [obj.settings.x0,obj.settings.y0,obj.settings.z0];
        space_beam = normpdf(x,pos_beam[1],obj.settings.sigmaX).*normpdf(y,pos_beam[2],obj.settings.sigmaY).*normpdf(z,pos_beam[3],obj.settings.sigmaZ);
        omega_beam = exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO2Inv*(obj.settings.Omega2-Omega[2])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2);
        return 10^5 .* omega_beam .* space_beam .* normpdf(E,obj.settings.eMax,obj.settings.sigmaE) .* obj.csd.S[n+1]
    elseif obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD"
        return 0.0;
    end
    return 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2)*exp(-sigmaEInv*(E0-E)^2)*exp(-sigmaXInv*(x-obj.settings.x0)^2)*exp(-sigmaYInv*(y-obj.settings.y0)^2)*obj.csd.S[n]*obj.settings.densityMin;
end

function BCLeft(obj::SolverCSD{T},n::Int) where {T<:AbstractFloat}
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

@inline minmod(x::T, y::T) where {T<:AbstractFloat} = ifelse(x < 0, clamp(y, x, 0.0), clamp(y, 0.0, x))

@inline function slopefit(left::T, center::T, right::T) where {T<:AbstractFloat}
    tmp = minmod(0.5 * (right - left),2.0 * (center - left));
    return minmod(2.0 * (right - center),tmp);
end

function solveFlux!(obj::SolverCSD{T}, phi::Array{T,4}, flux::Array{T,4}) where {T<:AbstractFloat}
    # computes the numerical flux over cell boundaries for each ordinate
    # for faster computation, we split the iteration over quadrature points
    # into four different blocks: North West, Nort East, Sout West, South East
    # this corresponds to the direction the ordinates point to
    idxPosPosPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxPosNegPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].<0.0) .&(obj.qReduced[:,3].>=0.0))
    idxNegPosPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxNegNegPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].<0.0) .&(obj.qReduced[:,3].>=0.0))

    idxPosPosNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxPosNegNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].<0.0) .&(obj.qReduced[:,3].<0.0))
    idxNegPosNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxNegNegNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].<0.0) .&(obj.qReduced[:,3].<0.0))

    nx = collect(3:(obj.settings.NCellsX-2));
    ny = collect(3:(obj.settings.NCellsY-2));
    nz = collect(3:(obj.settings.NCellsZ-2));
    

    # PosPos
    for j=ny,i=nx,k=nz, q = idxPosPosPos

        s1 = phi[i-2,j,k,q]
        s2 = phi[i-1,j,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i+1,j,k,q]
        eastflux = s3+0.5 .*slopefit(s2,s3,s4)
        westflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit(s2,s3,s4)
        southflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit(s2,s3,s4)
        downflux = s2+0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,2]./obj.settings.dy .* (northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end
    #PosNeg
    for j=ny,i=nx,k=nz,q = idxPosNegPos

        s1 = phi[i-2,j,k,q]
        s2 = phi[i-1,j,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i+1,j,k,q]
        eastflux = s3+0.5 .*slopefit(s2,s3,s4)
        westflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit(s2,s3,s4)
        southflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit(s2,s3,s4)
        downflux = s2+0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegPos
    for j=ny,i=nx,k=nz,q = idxNegPosPos
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit(s2,s3,s4)
        westflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit(s2,s3,s4)
        southflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit(s2,s3,s4)
        downflux = s2+0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegNeg
    for j=ny,i=nx,k=nz,q = idxNegNegPos
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit(s2,s3,s4)
        westflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit(s2,s3,s4)
        southflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit(s2,s3,s4)
        downflux = s2+0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # PosPos
    for j=ny,i=nx,k=nz, q = idxPosPosNeg
        s1 = phi[i-2,j,k,q]
        s2 = phi[i-1,j,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i+1,j,k,q]
        eastflux = s3+0.5 .*slopefit(s2,s3,s4)
        westflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit(s2,s3,s4)
        southflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit(s2,s3,s4)
        downflux = s2-0.5 .*slopefit(s1,s2,s3)

        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        
        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,2]./obj.settings.dy .* (northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end
    #PosNeg
    for j=ny,i=nx,k=nz,q = idxPosNegNeg
        s1 = phi[i-2,j,k,q]
        s2 = phi[i-1,j,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i+1,j,k,q]
        eastflux = s3+0.5 .*slopefit(s2,s3,s4)
        westflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit(s2,s3,s4)
        southflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit(s2,s3,s4)
        downflux = s2-0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegPos
    for j=ny,i=nx,k=nz,q = idxNegPosNeg
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit(s2,s3,s4)
        westflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit(s2,s3,s4)
        southflux = s2+0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit(s2,s3,s4)
        downflux = s2-0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegNeg
    for j=ny,i=nx,k=nz,q = idxNegNegNeg
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit(s2,s3,s4)
        westflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit(s2,s3,s4)
        southflux = s2-0.5 .*slopefit(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit(s2,s3,s4)
        downflux = s2-0.5 .*slopefit(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end
end

function SolveFirstCollisionSource(obj::SolverCSD{T}) where {T<:AbstractFloat}
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries

    # Set up initial condition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "protonBeam" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.MReduced = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

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
        dE = eTrafo[n]-eTrafo[n-1];
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
    
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFlux!(obj,psi,flux);

        psi .= psi .- dE*flux;
        
        psiNew .= psi ./ (1+dE*sigmaS[1]);

        # stream collided particles
        uTilde = u .- dE * obj.stencil.L2x*u*obj.pn.Ax - dE * obj.stencil.L2y*u*obj.pn.Az - dE * obj.stencil.L1x*u*obj.AbsAx - dE * obj.stencil.L1y*u*obj.AbsAz; 
        uTilde[obj.boundaryIdx,:] .= 0.0;

        # scatter particles
        for i = 2:(nx-1)
            for j = 2:(ny-1)
                idx = (i-1)*nx + j
                uNew[idx,:] = (Id .+ dE*D)\(uTilde[idx,:] .+ dE*Diagonal(Dvec)*obj.MReduced*psiNew[i,j,:]);
                uOUnc[idx] = psiNew[i,j,:]'*obj.MReduced[1,:];
            end
        end
        uNew[obj.boundaryIdx,:] .= 0.0;

        #uTilde[1,:] .= BCLeft(obj,n);
        #uNew = uTilde .- dE*uTilde*D;
        
        # update dose
        obj.dose .+= dE * (uNew[:,1]+uOUnc) * obj.csd.SMid[n-1] ./ obj.densityVec ./( 1 + (n==2||n==nEnergies));


        u .= uNew;
        u[obj.boundaryIdx,:] .= 0.0;
        psi .= psiNew;
        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose,psi;

end

function SolveFirstCollisionSourceDLR2ndOrder(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    # Set up initial condition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.M = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(nx*ny*nz,r));
    W,_,_ = svd!(zeros(N,r));

    # rank-r truncation:
    S = zeros(r,r);

    K = zeros(size(X));
    k1 = zeros(size(X));
    L = zeros(size(W));
    l1 = zeros(size(W));

    WAxW = zeros(r,r)
    WAyW = zeros(r,r)
    WAzW = zeros(r,r)

    XL2xX = zeros(r,r)
    XL2yX = zeros(r,r)
    XL2zX = zeros(r,r)

    MUp = zeros(r,r)
    NUp = zeros(r,r)

    XNew = zeros(nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    flux = zeros(size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(nx*ny*nz);
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    idx = vectorIndex(nx,ny,i,j,k)
                    uOUnc[idx] = psi[i,j,k,:]'*obj.M[1,:];
                end
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n-1] ./ obj.densityVec ;

        # stream uncollided particles
        solveFlux!(obj,psi./obj.density,flux);

        psiBC = psi[obj.boundaryIdx];

        psi .= (psi .- dE*flux) ./ (1+dE*sigmaS[1]);
        psi[obj.boundaryIdx] .= psiBC; # no scattering in boundary cells
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;
            K .= X*S;

            WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T
            WAyW .= W'*obj.pn.Ay*W # Ax  = Ax^T
            WAzW .= W'*obj.pn.Az*W # Az  = Az^T

            k1 .= -obj.stencil.L2x*K*WAxW .- obj.stencil.L2y*K*WAyW .- obj.stencil.L2z*K*WAzW;
            K .= K .+ dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+0.5*dE*k1)*WAxW .- obj.stencil.L2y*(K+0.5*dE*k1)*WAyW .- obj.stencil.L2z*(K+0.5*dE*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+0.5*dE*k1)*WAxW .- obj.stencil.L2y*(K+0.5*dE*k1)*WAyW .- obj.stencil.L2z*(K+0.5*dE*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+dE*k1)*WAxW .- obj.stencil.L2y*(K+dE*k1)*WAyW .- obj.stencil.L2z*(K+dE*k1)*WAzW;
            K .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            
            l1 .= -obj.pn.Ax*L*XL2xX' .- obj.pn.Ay*L*XL2yX' .- obj.pn.Az*L*XL2zX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+dE*l1)*XL2xX' .- obj.pn.Ay*(L+dE*l1)*XL2yX' .- obj.pn.Az*(L+dE*l1)*XL2zX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            # impose boundary condition
            #X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X

            WAxW .= W'*obj.pn.Ax*W
            WAyW .= W'*obj.pn.Ay*W
            WAzW .= W'*obj.pn.Az*W

            s1 = -XL2xX*S*WAxW .- XL2yX*S*WAyW .- XL2zX*S*WAzW;
            s2 = -XL2xX*(S+0.5*dE*s1)*WAxW .- XL2yX*(S+0.5*dE*s1)*WAyW .- XL2zX*(S+0.5*dE*s1)*WAzW;
            s3 = -XL2xX*(S+0.5*dE*s2)*WAxW .- XL2yX*(S+0.5*dE*s2)*WAyW .- XL2zX*(S+0.5*dE*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW .- XL2yX*(S+dE*s3)*WAyW .- XL2zX*(S+dE*s3)*WAzW;

            S .= S .+ dE .* (s1 .+ 2 * s2 .+ 2 * s3 .+ s4) ./ 6;
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,Sv,Tv = svd!(L);

        S .= Tv*Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * Ten2Vec(psi) * (obj.M' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.M*(Ten2Vec(psi)'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*Ten2Vec(psi))*obj.M'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    idx = vectorIndex(nx,ny,i,j,k)
                    uOUnc[idx] = psi[i,j,k,:]'*obj.M[1,:];
                end
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n] ./ obj.densityVec;

        #psi .= zeros(size(psi))
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V,W*V,obj.dose,psi;

end

function SolveFirstCollisionSourceDLR4thOrder(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = T.(obj.csd.eTrafo);
    energy = T.(obj.csd.eGrid);

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    x = obj.settings.xMid;
    y = obj.settings.yMid;
    z = obj.settings.zMid;

    sigmaO1Inv = 10000.0;
    sigmaO2Inv = 10000.0;
    sigmaO3Inv = 10000.0;
    pos_beam = [obj.settings.x0,obj.settings.y0,obj.settings.z0];

    # Set up initiandition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.M = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(T,N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(T,nx*ny*nz,r));
    W,_,_ = svd!(zeros(T,N,r));

    # rank-r truncation:
    S = zeros(T,r,r);

    K = zeros(T,size(X));
    k1 = zeros(T,size(X));
    L = zeros(T,size(W));
    l1 = zeros(T,size(W));

    WAxW = zeros(T,r,r)
    WAyW = zeros(T,r,r)
    WAzW = zeros(T,r,r)

    XL2xX = zeros(T,r,r)
    XL2yX = zeros(T,r,r)
    XL2zX = zeros(T,r,r)

    MUp = zeros(T,r,r)
    NUp = zeros(T,r,r)

    XNew = zeros(T,nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    psi = Ten2Vec(psi);

    prog = Progress(nEnergies-1,1)

    intSigma = dE * SigmaAtEnergy(obj.csd,energy[1])[1];
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        ############## Dose Computation ##############

        obj.dose .+= 0.5*dE * (X*S*W[1,:]+ psi * obj.M[1,:]) * obj.csd.S[n-1] ./ obj.densityVec ;

        intSigma += dE * sigmaS[1];
        for q = 1:nq
            beamOmega = 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-obj.qReduced[q,1])^2)*exp(-sigmaO2Inv*(obj.settings.Omega2-obj.qReduced[q,2])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-obj.qReduced[q,3])^2) * exp(-intSigma);
            for j = 1:ny
                beamy = normpdf(y[j] - eTrafo[n]*obj.qReduced[q,2],pos_beam[2],obj.settings.sigmaY)
                if beamy < 1e-6 continue; end
                for i = 1:nx
                    beamx = normpdf(x[i] - eTrafo[n]*obj.qReduced[q,1],pos_beam[1],obj.settings.sigmaX)
                    if beamx < 1e-6 continue; end
                    for k = 1:nz
                        beamz = normpdf(z[k] - eTrafo[n]*obj.qReduced[q,3],pos_beam[3],obj.settings.sigmaZ)
                        idx = vectorIndex(nx,ny,i,j,k)
                        psi[idx,q] = beamOmega * beamx * beamy * beamz             
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

        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;
            K .= X*S;

            WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T
            WAyW .= W'*obj.pn.Ay*W # Ax  = Ax^T
            WAzW .= W'*obj.pn.Az*W # Az  = Az^T

            k1 .= -obj.stencil.L2x*K*WAxW .- obj.stencil.L2y*K*WAyW .- obj.stencil.L2z*K*WAzW;
            K .= K .+ dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K.+(0.5*dE).*k1)*WAxW .- obj.stencil.L2y*(K.+(0.5*dE).*k1)*WAyW .- obj.stencil.L2z*(K.+(0.5*dE).*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K.+(0.5*dE).*k1)*WAxW .- obj.stencil.L2y*(K.+(0.5*dE).*k1)*WAyW .- obj.stencil.L2z*(K.+(0.5*dE).*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K.+dE.*k1)*WAxW .- obj.stencil.L2y*(K.+dE.*k1)*WAyW .- obj.stencil.L2z*(K.+dE.*k1)*WAzW;
            K .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            
            l1 .= -obj.pn.Ax*L*XL2xX' .- obj.pn.Ay*L*XL2yX' .- obj.pn.Az*L*XL2zX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+dE*l1)*XL2xX' .- obj.pn.Ay*(L+dE*l1)*XL2yX' .- obj.pn.Az*(L+dE*l1)*XL2zX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            # impose boundary condition
            #X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X

            WAxW .= W'*obj.pn.Ax*W
            WAyW .= W'*obj.pn.Ay*W
            WAzW .= W'*obj.pn.Az*W

            s1 = -XL2xX*S*WAxW .- XL2yX*S*WAyW .- XL2zX*S*WAzW;
            s2 = -XL2xX*(S+0.5*dE*s1)*WAxW .- XL2yX*(S+0.5*dE*s1)*WAyW .- XL2zX*(S+0.5*dE*s1)*WAzW;
            s3 = -XL2xX*(S+0.5*dE*s2)*WAxW .- XL2yX*(S+0.5*dE*s2)*WAyW .- XL2zX*(S+0.5*dE*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW .- XL2yX*(S+dE*s3)*WAyW .- XL2zX*(S+dE*s3)*WAzW;

            S .= S .+ dE .* (s1 .+ 2 * s2 .+ 2 * s3 .+ s4) ./ 6;
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,Sv,Tv = svd!(L);

        S .= Tv*Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * psi * (obj.M' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.M*(psi'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*psi)*obj.M'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+psi * obj.M[1,:]) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V,W*V,obj.dose,psi;

end

# only K-step computed with CUDA, not using characteristics
function CudaSolveFirstCollisionSourceDLR4thOrderSN(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = T.(obj.csd.eTrafo);
    energy = T.(obj.csd.eGrid);

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    x = obj.settings.xMid;
    y = obj.settings.yMid;
    z = obj.settings.zMid;

    sigmaO1Inv = 10000.0;
    sigmaO2Inv = 10000.0;
    sigmaO3Inv = 10000.0;
    pos_beam = [obj.settings.x0,obj.settings.y0,obj.settings.z0];

    # Set up initiandition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.M = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(T,N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(T,nx*ny*nz,r));
    W,_,_ = svd!(zeros(T,N,r));

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];

    # rank-r truncation:
    S = zeros(T,r,r);

    KC = CUDA.zeros(T,size(X));
    K = zeros(T,size(X));
    k1 = CUDA.zeros(T,size(X));
    L = zeros(T,size(W));
    l1 = zeros(T,size(W));

    WAxWC = CuArray(zeros(T,r,r))
    WAyWC = CuArray(zeros(T,r,r))
    WAzWC = CuArray(zeros(T,r,r))

    WAxW = zeros(T,r,r)
    WAyW = zeros(T,r,r)
    WAzW = zeros(T,r,r)

    XL2xX = zeros(T,r,r)
    XL2yX = zeros(T,r,r)
    XL2zX = zeros(T,r,r)

    MUp = zeros(T,r,r)
    NUp = zeros(T,r,r)

    L2x = CuSparseMatrixCSC(obj.stencil.L2x);
    L2y = CuSparseMatrixCSC(obj.stencil.L2y);
    L2z = CuSparseMatrixCSC(obj.stencil.L2z);

    XNew = zeros(T,nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    #psi = Ten2Vec(psi);
    flux = zeros(T, size(psi))
    uOUnc = zeros(T, nx * nz * ny)

    prog = Progress(nEnergies-1,1)

    intSigma = dE * SigmaAtEnergy(obj.csd,energy[1])[1];
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        ############## Dose Computation ##############

        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    idx = vectorIndex(nx,ny,i,j,k)
                    uOUnc[idx] = psi[i,j,k,:]'*obj.M[1,:];
                end
            end
        end

        obj.dose .+= 0.5*dE * (X*S*W[1,:] + uOUnc) * obj.csd.S[n-1] ./ obj.densityVec ;

        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        # stream uncollided particles
        solveFlux!(obj,psi,flux);

        psi .= psi .- dE*flux;

        psi .= psi ./ (1+dE*sigmaS[1]);

        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;
            KC .= CuArray(X*S);

            WAxWC .= CuArray(W'*obj.pn.Ax*W) # Ax  = Ax^T
            WAyWC .= CuArray(W'*obj.pn.Ay*W) # Ax  = Ax^T
            WAzWC .= CuArray(W'*obj.pn.Az*W) # Az  = Az^T

            dE12 = Float32(0.5*dE);
            dEf32 = Float32(dE);

            k1 .= -L2x*KC*WAxWC .- L2y*KC*WAyWC .- L2z*KC*WAzWC;
            KC .= KC .+ dE .* k1 ./ 6;
            k1 .= -L2x*(KC.+dE12.*k1)*WAxWC .- L2y*(KC.+dE12.*k1)*WAyWC .- L2z*(KC.+dE12.*k1)*WAzWC;
            KC .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(KC.+dE12.*k1)*WAxWC .- L2y*(KC.+dE12.*k1)*WAyWC .- L2z*(KC.+dE12.*k1)*WAzWC;
            KC .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(KC.+dEf32.*k1)*WAxWC .- L2y*(KC.+dEf32.*k1)*WAyWC .- L2z*(KC.+dEf32.*k1)*WAzWC;
            KC .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(Matrix(KC));

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            
            l1 .= -obj.pn.Ax*L*XL2xX' .- obj.pn.Ay*L*XL2yX' .- obj.pn.Az*L*XL2zX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+dE*l1)*XL2xX' .- obj.pn.Ay*(L+dE*l1)*XL2yX' .- obj.pn.Az*(L+dE*l1)*XL2zX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            # impose boundary condition
            #X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X

            WAxW .= W'*obj.pn.Ax*W
            WAyW .= W'*obj.pn.Ay*W
            WAzW .= W'*obj.pn.Az*W

            s1 = -XL2xX*S*WAxW .- XL2yX*S*WAyW .- XL2zX*S*WAzW;
            s2 = -XL2xX*(S+0.5*dE*s1)*WAxW .- XL2yX*(S+0.5*dE*s1)*WAyW .- XL2zX*(S+0.5*dE*s1)*WAzW;
            s3 = -XL2xX*(S+0.5*dE*s2)*WAxW .- XL2yX*(S+0.5*dE*s2)*WAyW .- XL2zX*(S+0.5*dE*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW .- XL2yX*(S+dE*s3)*WAyW .- XL2zX*(S+dE*s3)*WAzW;

            S .= S .+ dE .* (s1 .+ 2 * s2 .+ 2 * s3 .+ s4) ./ 6;
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,Sv,Tv = svd!(L);

        S .= Tv*Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * Ten2Vec(psi) * (obj.M' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.M*(Ten2Vec(psi)'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*Ten2Vec(psi))*obj.M'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+Ten2Vec(psi) * obj.M[1,:]) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V,W*V,obj.dose,psi;

end

# only K-step computed with CUDA, using characteristics
function CudaSolveFirstCollisionSourceDLR4thOrder(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = T.(obj.csd.eTrafo);
    energy = T.(obj.csd.eGrid);

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    x = obj.settings.xMid;
    y = obj.settings.yMid;
    z = obj.settings.zMid;

    sigmaO1Inv = 10000.0;
    sigmaO2Inv = 10000.0;
    sigmaO3Inv = 10000.0;
    pos_beam = [obj.settings.x0,obj.settings.y0,obj.settings.z0];

    # Set up initiandition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.M = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(T,N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(T,nx*ny*nz,r));
    W,_,_ = svd!(zeros(T,N,r));

    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];

    # rank-r truncation:
    S = zeros(T,r,r);

    KC = CUDA.zeros(T,size(X));
    K = zeros(T,size(X));
    k1 = CUDA.zeros(T,size(X));
    L = zeros(T,size(W));
    l1 = zeros(T,size(W));

    WAxWC = CuArray(zeros(T,r,r))
    WAyWC = CuArray(zeros(T,r,r))
    WAzWC = CuArray(zeros(T,r,r))

    WAxW = zeros(T,r,r)
    WAyW = zeros(T,r,r)
    WAzW = zeros(T,r,r)

    XL2xX = zeros(T,r,r)
    XL2yX = zeros(T,r,r)
    XL2zX = zeros(T,r,r)

    MUp = zeros(T,r,r)
    NUp = zeros(T,r,r)

    L2x = CuSparseMatrixCSC(obj.stencil.L2x);
    L2y = CuSparseMatrixCSC(obj.stencil.L2y);
    L2z = CuSparseMatrixCSC(obj.stencil.L2z);

    XNew = zeros(T,nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    psi = Ten2Vec(psi);

    prog = Progress(nEnergies-1,1)

    intSigma = dE * SigmaAtEnergy(obj.csd,energy[1])[1];
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        ############## Dose Computation ##############

        obj.dose .+= 0.5*dE * (X*S*W[1,:]+ psi * obj.M[1,:]) * obj.csd.S[n-1] ./ obj.densityVec ;

        intSigma += dE * sigmaS[1];
        for q = 1:nq
            beamOmega = 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-obj.qReduced[q,1])^2)*exp(-sigmaO2Inv*(obj.settings.Omega2-obj.qReduced[q,2])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-obj.qReduced[q,3])^2) * exp(-intSigma);
            for j = 1:ny
                beamy = normpdf(y[j] - eTrafo[n]*obj.qReduced[q,2],pos_beam[2],obj.settings.sigmaY)
                if beamy < 1e-6 continue; end
                for i = 1:nx
                    beamx = normpdf(x[i] - eTrafo[n]*obj.qReduced[q,1],pos_beam[1],obj.settings.sigmaX)
                    if beamx < 1e-6 continue; end
                    for k = 1:nz
                        beamz = normpdf(z[k] - eTrafo[n]*obj.qReduced[q,3],pos_beam[3],obj.settings.sigmaZ)
                        idx = vectorIndex(nx,ny,i,j,k)
                        psi[idx,q] = beamOmega * beamx * beamy * beamz             
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

        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;
            KC .= CuArray(X*S);

            WAxWC .= CuArray(W'*obj.pn.Ax*W) # Ax  = Ax^T
            WAyWC .= CuArray(W'*obj.pn.Ay*W) # Ax  = Ax^T
            WAzWC .= CuArray(W'*obj.pn.Az*W) # Az  = Az^T

            dE12 = Float32(0.5*dE);
            dEf32 = Float32(dE);

            k1 .= -L2x*KC*WAxWC .- L2y*KC*WAyWC .- L2z*KC*WAzWC;
            KC .= KC .+ dE .* k1 ./ 6;
            k1 .= -L2x*(KC.+dE12.*k1)*WAxWC .- L2y*(KC.+dE12.*k1)*WAyWC .- L2z*(KC.+dE12.*k1)*WAzWC;
            KC .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(KC.+dE12.*k1)*WAxWC .- L2y*(KC.+dE12.*k1)*WAyWC .- L2z*(KC.+dE12.*k1)*WAzWC;
            KC .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(KC.+dEf32.*k1)*WAxWC .- L2y*(KC.+dEf32.*k1)*WAyWC .- L2z*(KC.+dEf32.*k1)*WAzWC;
            KC .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(Matrix(KC));

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            
            l1 .= -obj.pn.Ax*L*XL2xX' .- obj.pn.Ay*L*XL2yX' .- obj.pn.Az*L*XL2zX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+dE*l1)*XL2xX' .- obj.pn.Ay*(L+dE*l1)*XL2yX' .- obj.pn.Az*(L+dE*l1)*XL2zX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            # impose boundary condition
            #X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X

            WAxW .= W'*obj.pn.Ax*W
            WAyW .= W'*obj.pn.Ay*W
            WAzW .= W'*obj.pn.Az*W

            s1 = -XL2xX*S*WAxW .- XL2yX*S*WAyW .- XL2zX*S*WAzW;
            s2 = -XL2xX*(S+0.5*dE*s1)*WAxW .- XL2yX*(S+0.5*dE*s1)*WAyW .- XL2zX*(S+0.5*dE*s1)*WAzW;
            s3 = -XL2xX*(S+0.5*dE*s2)*WAxW .- XL2yX*(S+0.5*dE*s2)*WAyW .- XL2zX*(S+0.5*dE*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW .- XL2yX*(S+dE*s3)*WAyW .- XL2zX*(S+dE*s3)*WAzW;

            S .= S .+ dE .* (s1 .+ 2 * s2 .+ 2 * s3 .+ s4) ./ 6;
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,Sv,Tv = svd!(L);

        S .= Tv*Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * psi * (obj.M' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.M*(psi'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*psi)*obj.M'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+psi * obj.M[1,:]) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V,W*V,obj.dose,psi;

end

function SetBCs!(obj::SolverCSD{T}, energy::T, n::Int, psiCPU::Array{T,2}) where {T<:AbstractFloat}
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = size(obj.qReduced, 1)
    # set boundary condition
    for q = 1:nq
        for j = 1:ny
            for k = 1:nz
                idx = vectorIndex(nx,ny,1,j,k)
                psiCPU[idx,q] = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[1],obj.settings.yMid[j],obj.settings.zMid[k],n-1);
                idx = vectorIndex(nx,ny,nx,j,k)
                psiCPU[idx,q] = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[end],obj.settings.yMid[j],obj.settings.zMid[k],n-1);
            end
        end
        for i = 1:nx
            for k = 1:nz
                idx = vectorIndex(nx,ny,i,1,k)
                psiCPU[idx,q] = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[1],obj.settings.zMid[k],n-1);
                idx = vectorIndex(nx,ny,i,ny,k)
                psiCPU[idx,q] = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[end],obj.settings.zMid[k],n-1);
            end
        end
        for i = 1:nx
            for j = 1:ny
                idx = vectorIndex(nx,ny,i,j,1)
                psiCPU[idx,q] = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[j],obj.settings.zMid[1],n-1);
                idx = vectorIndex(nx,ny,i,j,nz)
                psiCPU[idx,q] = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[j],obj.settings.zMid[end],n-1);
            end
        end
    end
end

function FindIdxBoundary(obj::SolverCSD{T}) where {T<:AbstractFloat}
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    energy = T.(obj.csd.eGrid);
    nq = size(obj.qReduced, 1)
    idxGrid = [];

    # set boundary condition
    for n = 1:length(energy)
        if normpdf(energy[n],obj.settings.eMax,obj.settings.sigmaE) < 1e-7
            break;
        end
        for q = 1:nq
            for j = 1:ny
                for k = 1:nz
                    idx = vectorIndex(nx,ny,1,j,k)
                    val = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[1],obj.settings.yMid[j],obj.settings.zMid[k],n);
                    if val > 1e-12
                        println(val)
                        idxGrid = Base.unique([idxGrid; idx]) 
                    end
                    idx = vectorIndex(nx,ny,nx,j,k)
                    val = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[end],obj.settings.yMid[j],obj.settings.zMid[k],n);
                    if val > 1e-12
                        println(val)
                        idxGrid = Base.unique([idxGrid; idx]) 
                    end
                end
            end
            for i = 1:nx
                for k = 1:nz
                    idx = vectorIndex(nx,ny,i,1,k)
                    val = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[1],obj.settings.zMid[k],n);
                    if val > 1e-12
                        println(val)
                        idxGrid = Base.unique([idxGrid; idx]) 
                    end
                    idx = vectorIndex(nx,ny,i,ny,k)
                    val = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[end],obj.settings.zMid[k],n);
                    if val > 1e-12
                        println(val)
                        idxGrid = Base.unique([idxGrid; idx]) 
                    end
                end
            end
            for i = 1:nx
                for j = 1:ny
                    idx = vectorIndex(nx,ny,i,j,1)
                    val = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[j],obj.settings.zMid[1],n);
                    if val > 1e-12
                        println(val)
                        idxGrid = Base.unique([idxGrid; idx]) 
                    end
                    idx = vectorIndex(nx,ny,i,j,nz)
                    val = PsiBeam(obj,obj.qReduced[q,:],energy[n],obj.settings.xMid[i],obj.settings.yMid[j],obj.settings.zMid[end],n);
                    if val > 1e-12
                        println(val)
                        idxGrid = Base.unique([idxGrid; idx]) 
                    end
                end
            end
        end
    end
    return idxGrid;
end

# full CUDA
function CudaFullSolveFirstCollisionSourceDLR4thOrder(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = T.(obj.csd.eTrafo);
    energy = T.(obj.csd.eGrid);

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    x = obj.settings.xMid;
    y = obj.settings.yMid;
    z = obj.settings.zMid;

    # setup spatial grid
    grid = zeros(nx*ny*nz,3)
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                idx = vectorIndex(nx,ny,i,j,k)
                grid[idx,1] = x[i];
                grid[idx,2] = y[j];
                grid[idx,3] = z[k];
            end
        end
    end

    sigmaO1Inv = 10000.0;
    sigmaO2Inv = 10000.0;
    sigmaO3Inv = 10000.0;
    pos_beam = [obj.settings.x0,obj.settings.y0,obj.settings.z0];

    # Set up initiandition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,T.(obj.Q.pointsxyz[k,:]),T(obj.settings.eMax),obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.M = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    weights = CuArray(T.(obj.Q.weights[idxBeam]));
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(T,N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(T,nx*ny*nz,r));
    W,_,_ = svd!(zeros(T,N,r));

    # rank-r truncation:
    X = CuArray(X[:,1:r]);
    W = CuArray(W[:,1:r]);

    # rank-r truncation:
    S = CUDA.zeros(T,r,r);

    K = CUDA.zeros(T,size(X));
    k1 = CUDA.zeros(T,size(X));
    L = CUDA.zeros(T,size(W));
    l1 = CUDA.zeros(T,size(W));

    WAxW = CUDA.zeros(T,r,r)
    WAyW = CUDA.zeros(T,r,r)
    WAzW = CUDA.zeros(T,r,r)

    XL2xX = CUDA.zeros(T,r,r)
    XL2yX = CUDA.zeros(T,r,r)
    XL2zX = CUDA.zeros(T,r,r)

    MUp = CUDA.zeros(T,r,r)
    NUp = CUDA.zeros(T,r,r)

    L2x = CuSparseMatrixCSC(obj.stencil.L2x);
    L2y = CuSparseMatrixCSC(obj.stencil.L2y);
    L2z = CuSparseMatrixCSC(obj.stencil.L2z);

    Ax = CuSparseMatrixCSC(obj.pn.Ax)
    Ay = CuSparseMatrixCSC(obj.pn.Ay)
    Az = CuSparseMatrixCSC(obj.pn.Az)

    XNew = CUDA.zeros(T,nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    psiCPU = Ten2Vec(psi)
    psi = CuArray(psiCPU);
    M1 = CuArray(obj.M[1,:])
    M = CuArray(obj.M)
    sPow = CuArray(obj.csd.S)
    densityVec = CuArray(obj.densityVec);
    dose = CuArray(obj.dose);

    prog = Progress(nEnergies-1,1)

    idxBeam = FindIdxBoundary(obj)

    intSigma = dE * SigmaAtEnergy(obj.csd,energy[1])[1];
    ∫Y₀⁰dΩ = T(4 * pi / sqrt(4 * pi)); 

    dE12 = T(0.5*dE);
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = CuArray(SigmaAtEnergy(obj.csd,energy[n]))# .* sqrt.(obj.gamma));

        # set boundary conditions in psiCPU
        #SetBCs!(obj, energy[n], n,psiCPU);

        ############## Dose Computation ##############
        
        dose .+= dE12 * (X*S*W[1,:] .* ∫Y₀⁰dΩ + psi * weights) * sPow[n-1] ./ densityVec ;

        intSigma += dE * sigmaS[1];

        # backward tracing when IC given
        psiCPU .= zeros(size(psiCPU))
        for q = 1:nq
            beamOmega = 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-obj.qReduced[q,1])^2)*exp(-sigmaO2Inv*(obj.settings.Omega2-obj.qReduced[q,2])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-obj.qReduced[q,3])^2) * exp(-intSigma);
            for j = 1:ny
                beamy = normpdf(y[j] - eTrafo[n]*obj.qReduced[q,2],pos_beam[2],obj.settings.sigmaY)
                if beamy < 1e-6 continue; end
                for i = 1:nx
                    beamx = normpdf(x[i] - eTrafo[n]*obj.qReduced[q,1],pos_beam[1],obj.settings.sigmaX)
                    if beamx < 1e-6 continue; end
                    for k = 1:nz
                        beamz = normpdf(z[k] - eTrafo[n]*obj.qReduced[q,3],pos_beam[3],obj.settings.sigmaZ)
                        idx = vectorIndex(nx,ny,i,j,k)
                        psiCPU[idx,q] = T(beamOmega * beamx * beamy * beamz)             
                    end
                end
            end
        end

        # forward tracing when BCs given
        #=for k = 1:length(idxBeam)
            x_val = grid[k,:] .+ eTrafo[n]*obj.qReduced[q,:]
        end=#

        psi .= CuArray(psiCPU)
       
        Dvec = CUDA.zeros(T,obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end
        
        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            K .= X*S;

            WAxW .= (Ax*W)'*W # Ax  = Ax^T
            WAyW .= (Ay*W)'*W # Ax  = Ax^T
            WAzW .= (Az*W)'*W # Az  = Az^T

            k1 .= -L2x*K*WAxW .- L2y*K*WAyW .- L2z*K*WAzW;
            K .= K .+ dE .* k1 ./ 6;
            k1 .= -L2x*(K.+dE12.*k1)*WAxW .- L2y*(K.+dE12.*k1)*WAyW .- L2z*(K.+dE12.*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(K.+dE12.*k1)*WAxW .- L2y*(K.+dE12.*k1)*WAyW .- L2z*(K.+dE12.*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(K.+dE.*k1)*WAxW .- L2y*(K.+dE.*k1)*WAyW .- L2z*(K.+dE.*k1)*WAzW;
            K .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*(L2x*X)
            XL2yX .= X'*(L2y*X)
            XL2zX .= X'*(L2z*X)
            
            l1 .= -Ax*L*XL2xX' .- Ay*L*XL2yX' .- Az*L*XL2zX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -Ax*(L+dE12*l1)*XL2xX' .- Ay*(L+dE12*l1)*XL2yX' .- Az*(L+dE12*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -Ax*(L+dE12*l1)*XL2xX' .- Ay*(L+dE12*l1)*XL2yX' .- Az*(L+dE12*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -Ax*(L+dE*l1)*XL2xX' .- Ay*(L+dE*l1)*XL2yX' .- Az*(L+dE*l1)*XL2zX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*(L2x*X)
            XL2yX .= X'*(L2y*X)
            XL2zX .= X'*(L2z*X)

            WAxW .= W'*(Ax*W)
            WAyW .= W'*(Ay*W)
            WAzW .= W'*(Az*W)

            s1 = -XL2xX*S*WAxW .- XL2yX*S*WAyW .- XL2zX*S*WAzW;
            s2 = -XL2xX*(S+dE12*s1)*WAxW .- XL2yX*(S+dE12*s1)*WAyW .- XL2zX*(S+dE12*s1)*WAzW;
            s3 = -XL2xX*(S+dE12*s2)*WAxW .- XL2yX*(S+dE12*s2)*WAyW .- XL2zX*(S+dE12*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW .- XL2yX*(S+dE*s3)*WAyW .- XL2zX*(S+dE*s3)*WAzW;

            S .= S .+ dE .* (s1 .+ 2 * s2 .+ 2 * s3 .+ s4) ./ 6;
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] ./= (1 .+ dE*(sigmaS[1] .- Dvec))
        end

        W,Sv,Tv = svd!(L);

        S .= Tv*Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        K .= X*S;
        K .= K .+ dE * psi * (M' * (Diagonal(Dvec) * W) );

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*M*(psi'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*psi)*M'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        dose .+= dE12 * (X*S*W[1,:] * ∫Y₀⁰dΩ + psi * weights) * sPow[n] ./ densityVec ;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(Matrix(S));

    # return solution and dose
    return Matrix(X)*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*Matrix(W)*V,Matrix(W)*V,Vector(dose),Matrix(psi);

end

function SolveFirstCollisionSourceDLR4thOrderFP(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    x = obj.settings.xMid;
    y = obj.settings.yMid;
    z = obj.settings.zMid;

    sigmaO1Inv = 10000.0;
    sigmaO2Inv = 10000.0;
    sigmaO3Inv = 10000.0;
    sigmaEInv = 1000.0;
    densityMin = 1.0;
    pos_beam = [obj.settings.x0,obj.settings.y0,obj.settings.z0];

    # Set up initiandition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.M = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(nx*ny*nz,r));
    W,_,_ = svd!(zeros(N,r));

    # rank-r truncation:
    S = zeros(r,r);

    K = zeros(size(X));
    k1 = zeros(size(X));
    L = zeros(size(W));
    l1 = zeros(size(W));

    WAxW = zeros(r,r)
    WAyW = zeros(r,r)
    WAzW = zeros(r,r)

    XL2xX = zeros(r,r)
    XL2yX = zeros(r,r)
    XL2zX = zeros(r,r)

    MUp = zeros(r,r)
    NUp = zeros(r,r)

    XNew = zeros(nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    psi = Ten2Vec(psi);
    psi1 = zeros(size(psi));

    prog = Progress(nEnergies-1,1)

    intSigma = dE * SigmaAtEnergy(obj.csd,energy[1])[1];
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        ############## Dose Computation ##############

        obj.dose .+= 0.5*dE * (X*S*W[1,:]+ psi * obj.M[1,:]) * obj.csd.S[n-1] ./ obj.densityVec ;

        intSigma += dE * sigmaS[1];

        print("ray-trace... ")
        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    for q = 1:nq
                        idx = vectorIndex(nx,ny,i,j,k)
                        psi[idx,q] = PsiBeam(obj,obj.qReduced[q,:],0.0,x[i] - eTrafo[n]*obj.qReduced[q,1],y[j] - eTrafo[n]*obj.qReduced[q,2],z[k] - eTrafo[n]*obj.qReduced[q,3],1) * exp(-intSigma);
                    end
                end
            end
        end
        println("DONE.")
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        if n > 2 # perform streaming update after first collision (before solution is zero)
            print("K stream... ")
            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;
            K .= X*S;

            WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T
            WAyW .= W'*obj.pn.Ay*W # Ax  = Ax^T
            WAzW .= W'*obj.pn.Az*W # Az  = Az^T

            k1 .= -obj.stencil.L2x*K*WAxW .- obj.stencil.L2y*K*WAyW .- obj.stencil.L2z*K*WAzW;
            K .= K .+ dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+0.5*dE*k1)*WAxW .- obj.stencil.L2y*(K+0.5*dE*k1)*WAyW .- obj.stencil.L2z*(K+0.5*dE*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+0.5*dE*k1)*WAxW .- obj.stencil.L2y*(K+0.5*dE*k1)*WAyW .- obj.stencil.L2z*(K+0.5*dE*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+dE*k1)*WAxW .- obj.stencil.L2y*(K+dE*k1)*WAyW .- obj.stencil.L2z*(K+dE*k1)*WAzW;
            K .+= dE .* k1 ./ 6;

            XNew,S1,S2 = svd!(K);
            Sk = Diagonal(S1)*S2'

            MUp .= XNew' * X;
            ################## L-step ##################
            print("L stream... ")
            L .= W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            
            l1 .= -obj.pn.Ax*L*XL2xX' .- obj.pn.Ay*L*XL2yX' .- obj.pn.Az*L*XL2zX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' .- obj.pn.Ay*(L+0.5*dE*l1)*XL2yX' .- obj.pn.Az*(L+0.5*dE*l1)*XL2zX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -obj.pn.Ax*(L+dE*l1)*XL2xX' .- obj.pn.Ay*(L+dE*l1)*XL2yX' .- obj.pn.Az*(L+dE*l1)*XL2zX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,S1,S2 = svd!(L);
            Sl = S2*Diagonal(S1)

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            ################## S-step ##################
            S .= 0.5 * (Sk*NUp' .+ MUp*Sl);
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,Sv,Tv = svd!(L);

        S .= Tv*Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * psi * (obj.M' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,S1,S2 = svd!(K);
        Sk = Diagonal(S1)*S2'

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.M*(psi'*X);

        WNew,S1,S2 = svd!(L);
        Sl = S2*Diagonal(S1)

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= 0.5 * (Sk*NUp' .+ MUp*Sl);

        ############## Dose Computation ##############
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+psi * obj.M[1,:]) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V,W*V,obj.dose,psi;

end

function SolveFirstCollisionSourceDLR(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    # Set up initial condition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,obj.settings.z0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    obj.MReduced = obj.M[:,idxBeam]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);

    # define density matrix
    Id = Diagonal(ones(N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(nx*ny*nz,r));
    W,_,_ = svd!(zeros(N,r));

    # rank-r truncation:
    S = zeros(r,r);
    K = zeros(size(X));

    WAxW = zeros(r,r)
    WAyW = zeros(r,r)
    WAzW = zeros(r,r)
    WAbsAxW = zeros(r,r)
    WAbsAyW = zeros(r,r)
    WAbsAzW = zeros(r,r)

    XL2xX = zeros(r,r)
    XL2yX = zeros(r,r)
    XL2zX = zeros(r,r)
    XL1xX = zeros(r,r)
    XL1yX = zeros(r,r)
    XL1zX = zeros(r,r)

    MUp = zeros(r,r)
    NUp = zeros(r,r)

    XNew = zeros(nx*ny*nz,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)/minimum(obj.density))

    flux = zeros(size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(nx*ny*nz);
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    idx = vectorIndex(nx,ny,i,j,k)
                    uOUnc[idx] = psi[i,j,k,:]'*obj.MReduced[1,:];
                end
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n-1] ./ obj.densityVec ;

        # stream uncollided particles
        solveFlux!(obj,psi./obj.density,flux);

        psiBC = psi[obj.boundaryIdx];

        psi .= (psi .- dE*flux) ./ (1+dE*sigmaS[1]);
        psi[obj.boundaryIdx] .= psiBC; # no scattering in boundary cells
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end

        D = Diagonal(sigmaS[1] .- Dvec);

        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;
            K .= X*S;

            WAxW .= W'*obj.pn.Ax*W
            WAyW .= W'*obj.pn.Ay*W
            WAzW .= W'*obj.pn.Az*W
            
            WAbsAzW .= W'*obj.AbsAz*W
            WAbsAyW .= W'*obj.AbsAy*W
            WAbsAxW .= W'*obj.AbsAx*W
            
            K .= K .- dE*(obj.stencil.L2x*K*WAxW + obj.stencil.L2y*K*WAyW + obj.stencil.L2z*K*WAzW + obj.stencil.L1x*K*WAbsAxW + obj.stencil.L1y*K*WAbsAyW + obj.stencil.L1z*K*WAbsAzW);

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L = W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            XL1xX .= X'*obj.stencil.L1x*X
            XL1yX .= X'*obj.stencil.L1y*X
            XL1zX .= X'*obj.stencil.L1z*X

            L .= L .- dE*(obj.pn.Ax*L*XL2xX' + obj.pn.Ay*L*XL2yX' + obj.pn.Az*L*XL2zX' + obj.AbsAx*L*XL1xX' + obj.AbsAy*L*XL1yX' + obj.AbsAz*L*XL1zX');
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;

            W .= WNew;
            X .= XNew;

            # impose boundary condition
            #X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL2zX .= X'*obj.stencil.L2z*X
            XL1xX .= X'*obj.stencil.L1x*X
            XL1yX .= X'*obj.stencil.L1y*X
            XL1zX .= X'*obj.stencil.L1z*X

            WAxW .= W'*obj.pn.Ax*W
            WAyW .= W'*obj.pn.Ay*W
            WAzW .= W'*obj.pn.Az*W
            
            WAbsAxW .= W'*obj.AbsAx*W
            WAbsAyW .= W'*obj.AbsAy*W
            WAbsAzW .= W'*obj.AbsAz*W

            S .= S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAyW + XL2zX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAyW + XL1zX*S*WAbsAzW);

            #if n > 20 break; end
        end

        ############## Out Scattering ##############
        L = W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,S1,S2 = svd!(L)
        S .= S2 * Diagonal(S1)

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * Ten2Vec(psi) * (obj.MReduced' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.MReduced*(Ten2Vec(psi)'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*Ten2Vec(psi))*obj.MReduced'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                for k = 1:nz
                    idx = vectorIndex(nx,ny,i,j,k)
                    uOUnc[idx] = psi[i,j,k,:]'*obj.MReduced[1,:];
                end
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V, W*V,obj.dose,psi;

end

function Solve(obj::SolverCSD{T}) where {T<:AbstractFloat}
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

    # Set up initial condition and store as matrix
    v = SetupICMoments(obj);
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    N = obj.pn.nTotalEntries
    u = zeros(nx*ny,N);
    for k = 1:N
        u[:,k] = vec(v[:,:,k]);
    end

    Id = Diagonal(ones(N));

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(1.0 ./obj.density))

    uNew = deepcopy(u)

    prog = Progress(nEnergies-1,1)

    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);#1000.0.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)
       
        Dvec = zeros(obj.pn.nTotalEntries)
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                Dvec[i+1] = sigmaS[l+1]
            end
        end
        #ET = expm1div.(-(sigmaS[1] .- Dvec)*dE);

        D = Diagonal((sigmaS[1] .- Dvec));#.*ET);
        #println("E = ",energy[n]," ",sigmaS[1])
        #println("ETrafo = ",eTrafo[n]," ",sigmaS[1].-sigmaS)

        # perform time update
        uTilde = u .- dE * RhsOld(obj,u);

        for j = 1:size(uNew,1)
            uNew[j,:] = (Id .+ dE*D)\uTilde[j,:];
        end
        #uNew[obj.boundaryIdx] .= u[obj.boundaryIdx]; # no scattering in boundary cells
        
        # update dose
        #println("weight = ", obj.csd.SMid[n-1] ./( 1 + (n==2||n==nEnergies)))
        obj.dose .+= dE * uNew[:,1] * obj.csd.SMid[n-1] ./( 1 + (n==2||n==nEnergies));
        #obj.dose .+= dE * uNew[:,1] * obj.csd.SMid[n-1] ./( 1 + (n==2||n==nEnergies));

        u .= uNew;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose./ obj.densityVec;

end
