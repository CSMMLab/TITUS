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

include("CSD.jl")
include("PNSystem.jl")
include("quadratures/Quadrature.jl")
include("utils.jl")
include("stencils.jl")

mutable struct SolverCSD{T<:AbstractFloat}
    # spatial grid of cell interfaces
    x::Array{T};
    y::Array{T};

    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{T,1};
    # Roe matrix
    AbsAx::SparseMatrixCSC{T, Int64};
    AbsAz::SparseMatrixCSC{T, Int64};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # stencil matrices
    stencil::Stencils;

    # material density
    density::Array{T,2};
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
        Ax,_,Az = SetupSystemMatrices(pn);
        SetupSystemMatricesSparse(pn);

        # setup Roe matrix
        S = eigvals(Ax)
        V = eigvecs(Ax)
        AbsAx = V*abs.(Diagonal(S))*inv(V)

        idx = findall(abs.(AbsAx) .> 1e-10)
        Ix = first.(Tuple.(idx)); Jx = last.(Tuple.(idx)); vals = AbsAx[idx];
        AbsAx = sparse(Ix,Jx,T.(vals),pn.nTotalEntries,pn.nTotalEntries);
        
        S = eigvals(Az)
        V = eigvecs(Az)
        AbsAz = V*abs.(diagm(S))*inv(V)
        idx = findall(abs.(AbsAz) .> 1e-10)
        Iz = first.(Tuple.(idx)); Jz = last.(Tuple.(idx)); valsz = AbsAz[idx];
        AbsAz = sparse(Iz,Jz,T.(valsz),pn.nTotalEntries,pn.nTotalEntries);

        # set density vector
        density = T.(settings.density);

        # allocate dose vector
        dose = zeros(T, settings.NCellsX*settings.NCellsY)

        nx = settings.NCellsX;
        ny = settings.NCellsY;

        # collect boundary indices
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

        # setup quadrature
        qorder = settings.nPN+1; # must be even for standard quadrature
        qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
        Q = Quadrature(qorder,qtype)

        Norder = pn.nTotalEntries
        O,M = ComputeTrafoMatrices(Q,Norder,settings.nPN)

        stencil = Stencils(settings,T);

        densityVec = vec(Mat2Vec(density));

        new{T}(T.(x),T.(y),settings,gamma,AbsAx,AbsAz,csd,pn,stencil,density,T.(densityVec),dose,boundaryIdx,Q,T.(O),T.(M),T);
    end
end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""

function SetupIC(obj::SolverCSD{T},pointsxyz::Matrix{Float64}) where {T<:AbstractFloat}
    nq = size(pointsxyz)[1];
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    psi = zeros(T, obj.settings.NCellsX,obj.settings.NCellsY,nq);
    pos_beam = [obj.settings.x0, obj.settings.y0,0];
    sigmaO1Inv = 10000.0;
    sigmaO3Inv = 10000.0;

    if obj.settings.problem == "validation" || obj.settings.problem == "waterBeam" 
        for i = 3:nx - 2
            for j = 3:ny - 2
                if obj.settings.problem == "waterBeam"
                    space_beam = normpdf(obj.settings.xMid[i],pos_beam[1],.1).*normpdf(obj.settings.yMid[j],pos_beam[2],.1);
                else
                    space_beam = normpdf(obj.settings.xMid[i],pos_beam[1],.4).*normpdf(obj.settings.yMid[j],pos_beam[2],.4);
                end
                for k = 1:nq 
                    #trafo = obj.csd.S[1]*obj.settings.density[i,j]; 
                    psi[i,j,k] = 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-pointsxyz[k,1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-pointsxyz[k,3])^2)*space_beam#*trafo;
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

function PsiLeft(obj::SolverCSD{T},n::Int,mu::Float64) where {T<:AbstractFloat}
    E0 = obj.settings.eMax;
    return 10^5*exp(-200.0*(1.0-mu)^2)*exp(-50*(E0-E)^2)
end

function PsiBeam(obj::SolverCSD{T},Omega::Array{T,1},E::T,x::Float64,y::Float64,n::Int) where {T<:AbstractFloat}
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
        sigmaO3Inv = 10000.0;
        sigmaEInv = 1000.0;
        densityMin = 1.0;
        pos_beam = [obj.settings.x0,obj.settings.y0,0];
        space_beam = normpdf(x,pos_beam[1],.1).*normpdf(y,pos_beam[2],.1);
        #println(space_beam)
        return 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2)*space_beam*obj.csd.S[n]*densityMin#*exp(-sigmaEInv*(E0-E)^2)#;
    elseif obj.settings.problem == "waterBeam" || obj.settings.problem == "waterBeamElectrons"
        sigmaO1Inv = 10000.0;
        sigmaO3Inv = 10000.0;
        sigmaEInv = 1000.0;
        densityMin = 1.0;
        pos_beam = [obj.settings.x0,obj.settings.y0,0];
        space_beam = normpdf(x,pos_beam[1],.1).*normpdf(y,pos_beam[2],.1);
        #println(space_beam)
        return 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2)*space_beam*obj.csd.S[n]*densityMin#*exp(-sigmaEInv*(E0-E)^2)#;
    elseif obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD"
        return 0.0;
    end
    return 10^5*exp(-sigmaO1Inv*(obj.settings.Omega1-Omega[1])^2)*exp(-sigmaO3Inv*(obj.settings.Omega3-Omega[3])^2)*exp(-sigmaEInv*(E0-E)^2)*exp(-sigmaXInv*(x-obj.settings.x0)^2)*exp(-sigmaYInv*(y-obj.settings.y0)^2)*obj.csd.S[n]*obj.settings.densityMin;
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

function solveFlux!(obj::SolverCSD{T}, phi::Array{T,3}, flux::Array{T,3}) where {T<:AbstractFloat}
    # computes the numerical flux over cell boundaries for each ordinate
    # for faster computation, we split the iteration over quadrature points
    # into four different blocks: North West, Nort East, Sout West, South East
    # this corresponds to the direction the ordinates point to
    idxPosPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxPosNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxNegPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,3].>=0.0))
    idxNegNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,3].<0.0))

    nx = collect(3:(obj.settings.NCellsX-2));
    ny = collect(3:(obj.settings.NCellsY-2));

    # PosPos
    for j=ny,i=nx, q = idxPosPos
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

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,3]./obj.settings.dy .* (northflux-southflux)
    end
    #PosNeg
    for j=ny,i=nx,q = idxPosNeg
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

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegPos
    for j=ny,i=nx,q = idxNegPos
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

        flux[i,j,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end

    # NegNeg
    for j=ny,i=nx,q = idxNegNeg
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

        flux[i,j,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,3] ./obj.settings.dy .*(northflux-southflux)
    end
end

function solveFluxUpwind!(obj::SolverCSD{T}, phi::Array{T,3}, flux::Array{T,3}) where {T<:AbstractFloat}
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

function SolveFirstCollisionSource(obj::SolverCSD{T}) where {T<:AbstractFloat}
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;

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
    elseif obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" || obj.settings.problem == "waterBeam" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,T.(obj.Q.pointsxyz[k,:]),T(obj.settings.eMax),obj.settings.x0,obj.settings.y0,1)
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
    Id = Diagonal(ones(T,N));

    # setup gamma vector (square norm of P) to nomralize
    settings = obj.settings

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/obj.settings.dx*maximum(densityInv))

    u = zeros(nx*ny,N);
    uNew = deepcopy(u)
    flux = zeros(T,size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(T,nx*ny);

    psiNew = zeros(T,size(psi));
    uTilde = zeros(T,size(u))

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

function SolveFirstCollisionSourceDLRHighOrder(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    counterPNG = 0

    # Set up initial condition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    else #if obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,obj.Q.pointsxyz[k,:],obj.settings.eMax,obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    nq = length(idxBeam);

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(nx*ny,r));
    W,_,_ = svd!(zeros(N,r));
    
    # rank-r truncation:
    X = Matrix(X[:,1:r]);
    W = Matrix(W[:,1:r]);
    S = zeros(r,r);
    K = zeros(size(X));
    k1 = zeros(size(X));
    L = zeros(size(W));

    WAxW = zeros(r,r);
    WAzW = zeros(r,r);

    XL2xX = zeros(r,r);
    XL2yX = zeros(r,r);
    XL1xX = zeros(r,r);
    XL1yX = zeros(r,r);

    MUp = zeros(r,r);
    NUp = zeros(r,r);

    XNew = zeros(nx*ny,r);
    STmp = zeros(r,r);

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)*maximum(densityInv))

    flux = zeros(size(psi));

    prog = Progress(nEnergies-1,1);

    uOUnc = zeros(nx*ny);

    XX = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
    YY = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        # set boundary condition
        if obj.settings.problem != "validation" ||  obj.settings.problem != "waterBeam" # validation testcase sets beam in initial condition
            for k = 1:nq
                for j = 1:nx
                    psi[j,1,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[1],n-1);
                    psi[j,end,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[end],n-1);
                    psi[j,2,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[1],n-1);
                    psi[j,end - 1,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[end],n-1);
                end
                for j = 1:ny
                    psi[1,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[1],obj.settings.yMid[j],n-1);
                    psi[end,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[end],obj.settings.yMid[j],n-1);
                    psi[2,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[1],obj.settings.yMid[j],n-1);
                    psi[end - 1,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[end],obj.settings.yMid[j],n-1);
                end
            end
        end

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                uOUnc[idx] = psi[i,j,:]'*obj.MReduced[1,:];
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

            WAzW .= W'*obj.pn.Az*W # Az  = Az^T
            WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T

            k1 .= -obj.stencil.L2x*K*WAxW - obj.stencil.L2y*K*WAzW;
            K .= K .+ dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+0.5*dE*k1)*WAxW - obj.stencil.L2y*(K+0.5*dE*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+0.5*dE*k1)*WAxW - obj.stencil.L2y*(K+0.5*dE*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -obj.stencil.L2x*(K+dE*k1)*WAxW - obj.stencil.L2y*(K+dE*k1)*WAzW;
            K .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL1xX .= X'*obj.stencil.L1x*X
            XL1yX .= X'*obj.stencil.L1y*X

            l1 = -obj.pn.Ax*L*XL2xX' - obj.pn.Az*L*XL2yX';
            l2 = -obj.pn.Ax*(L+0.5*dE*l1)*XL2xX' - obj.pn.Az*(L+0.5*dE*l1)*XL2yX';
            l3 = -obj.pn.Ax*(L+0.5*dE*l2)*XL2xX' - obj.pn.Az*(L+0.5*dE*l2)*XL2yX';
            l4 = -obj.pn.Ax*(L+dE*l3)*XL2xX' - obj.pn.Az*(L+dE*l3)*XL2yX';

            L .= L .+ dE .* (l1 .+ 2 * l2 .+ 2 * l3 .+ l4) ./ 6;
                    
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
            XL1xX .= X'*obj.stencil.L1x*X
            XL1yX .= X'*obj.stencil.L1y*X

            WAzW .= W'*obj.pn.Az*W
            WAxW .= W'*obj.pn.Ax*W

            s1 = -XL2xX*S*WAxW - XL2yX*S*WAzW;
            s2 = -XL2xX*(S+0.5*dE*s1)*WAxW - XL2yX*(S+0.5*dE*s1)*WAzW;
            s3 = -XL2xX*(S+0.5*dE*s2)*WAxW - XL2yX*(S+0.5*dE*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW - XL2yX*(S+dE*s3)*WAzW;

            S .= S .+ dE .* (s1 .+ 2 * s2 .+ 2 * s3 .+ s4) ./ 6;
        end

        ############## Out Scattering ##############
        L .= W*S';

        for i = 1:r
            L[:,i] = (Id .+ dE*D)\L[:,i]
        end

        W,Sv,Ts = svd!(L);

        S .= Ts * Diagonal(Sv);

        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K .= X*S;
        #u = u .+dE*Mat2Vec(psiNew)*M'*Diagonal(Dvec);
        K .= K .+ dE * Mat2Vec(psi) * (obj.MReduced' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.MReduced*(Mat2Vec(psi)'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*Mat2Vec(psi))*obj.MReduced'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                uOUnc[idx] = psi[i,j,:]'*obj.MReduced[1,:];
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n] ./ obj.densityVec;

        ######## Plotting #########
        if mod(n-1,Int(round(7*nEnergies/474/7))) == 0
            counterPNG += 1;
            
            U,Sigma,V = svd(S);
            close("all")
            fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
            ax = gca()
            pcolormesh(YY,XX,Vec2Mat(s.NCellsX,s.NCellsY,uOUnc)[2:end-1,2:end-1]')
            ax.tick_params("both",labelsize=20) 
            plt.xlabel("x", fontsize=20)
            plt.ylabel("y", fontsize=20)
            plt.title("scalar flux,  E = $(round(obj.csd.eGrid[n], digits=3)) MeV", fontsize=25)
            tight_layout()
            if counterPNG < 10
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_000$(counterPNG).png")
            elseif counterPNG < 100
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_00$(counterPNG).png")
            elseif counterPNG < 1000
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_0$(counterPNG).png")
            else
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_$(counterPNG).png")
            end

            close("all")
            fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
            ax = gca()
            pcolormesh(YY,XX,Vec2Mat(s.NCellsX,s.NCellsY,X*U[:,1])[2:end-1,2:end-1]')
            ax.tick_params("both",labelsize=20) 
            plt.xlabel("x", fontsize=20)
            plt.ylabel("y", fontsize=20)
            plt.title("dominant spatial mode,  E = $(round(obj.csd.eGrid[n], digits=3)) MeV", fontsize=25)
            tight_layout()
            if counterPNG < 10
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_000$(counterPNG).png")
            elseif counterPNG < 100
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_00$(counterPNG).png")
            elseif counterPNG < 1000
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_0$(counterPNG).png")
            else
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_$(counterPNG).png")
            end
            
            # write modal basis
            writedlm("output/gifW/W_$(counterPNG)",W*V)
            #break
        end

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
    else#if obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,T.(obj.Q.pointsxyz[k,:]),T(obj.settings.eMax),obj.settings.x0,obj.settings.y0,1)
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
    densityInv = Diagonal(1.0 ./obj.density);
    Id = Diagonal(ones(T,N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(T,nx*ny,r));
    W,_,_ = svd!(zeros(T,N,r));
    
    # rank-r truncation:
    X = Matrix(X[:,1:r]);
    W = Matrix(W[:,1:r]);
    S = zeros(T,r,r);
    K = zeros(T,size(X));

    WAxW = zeros(T,r,r)
    WAzW = zeros(T,r,r)
    WAbsAxW = zeros(T,r,r)
    WAbsAzW = zeros(T,r,r)

    XL2xX = zeros(T,r,r)
    XL2yX = zeros(T,r,r)
    XL1xX = zeros(T,r,r)
    XL1yX = zeros(T,r,r)

    MUp = zeros(T,r,r)
    NUp = zeros(T,r,r)

    XNew = zeros(T,nx*ny,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)*maximum(densityInv))

    flux = zeros(T,size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(T,nx*ny);
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if obj.settings.problem != "validation" # validation testcase sets beam in initial condition
            for k = 1:nq
                for j = 1:nx
                    psi[j,1,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[1],n-1);
                    psi[j,end,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[end],n-1);
                end
                for j = 1:ny
                    psi[1,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[1],obj.settings.yMid[j],n-1);
                    psi[end,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[end],obj.settings.yMid[j],n-1);
                end
            end
        end

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                uOUnc[idx] = psi[i,j,:]'*obj.MReduced[1,:];
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n-1] ./ obj.densityVec ;

        # stream uncollided particles
        solveFluxUpwind!(obj,psi./obj.density,flux);

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

            WAzW .= W'*obj.pn.Az*W # Az  = Az^T
            WAbsAzW .= W'*obj.AbsAz*W
            WAbsAxW .= W'*obj.AbsAx*W
            WAxW .= W'*obj.pn.Ax*W # Ax  = Ax^T

            K .= K .- dE*(obj.stencil.L2x*K*WAxW + obj.stencil.L2y*K*WAzW + obj.stencil.L1x*K*WAbsAxW + obj.stencil.L1y*K*WAbsAzW);

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L = W*S';

            XL2xX .= X'*obj.stencil.L2x*X
            XL2yX .= X'*obj.stencil.L2y*X
            XL1xX .= X'*obj.stencil.L1x*X
            XL1yX .= X'*obj.stencil.L1y*X

            L .= L .- dE*(obj.pn.Ax*L*XL2xX' + obj.pn.Az*L*XL2yX' + obj.AbsAx*L*XL1xX' + obj.AbsAz*L*XL1yX');
                    
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
            XL1xX .= X'*obj.stencil.L1x*X
            XL1yX .= X'*obj.stencil.L1y*X

            WAzW .= W'*obj.pn.Az*W
            WAbsAzW .= W'*obj.AbsAz*W
            WAbsAxW .= W'*obj.AbsAx*W
            WAxW .= W'*obj.pn.Ax*W

            S .= S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAzW);
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
        K .= K .+ dE * Mat2Vec(psi) * (obj.MReduced' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.MReduced*(Mat2Vec(psi)'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*Mat2Vec(psi))*obj.MReduced'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                uOUnc[idx] = psi[i,j,:]'*obj.MReduced[1,:];
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V, W*V,obj.dose,psi;

end

function CSolveFirstCollisionSourceDLR(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
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

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(nx*ny,r));
    W,_,_ = svd!(zeros(N,r));
    
    # rank-r truncation:
    X = CuArray(X[:,1:r]);
    W = CuArray(W[:,1:r]);
    S = CUDA.zeros(r,r);
    K = CUDA.zeros(size(X));

    WAxW = CUDA.zeros(r,r)
    WAzW = CUDA.zeros(r,r)
    WAbsAxW = CUDA.zeros(r,r)
    WAbsAzW = CUDA.zeros(r,r)

    XL2xX = CUDA.zeros(r,r)
    XL2yX = CUDA.zeros(r,r)
    XL1xX = CUDA.zeros(r,r)
    XL1yX = CUDA.zeros(r,r)

    MUp = CUDA.zeros(r,r)
    NUp = CUDA.zeros(r,r)

    XNew = CUDA.zeros(nx*ny,r)

    Ax = CuSparseMatrixCSC(obj.pn.Ax);
    Az = CuSparseMatrixCSC(obj.pn.Az);
    AbsAx = CuSparseMatrixCSC(obj.AbsAx);
    AbsAz = CuSparseMatrixCSC(obj.AbsAz);

    L1x = CuSparseMatrixCSC(obj.stencil.L1x)
    L1y = CuSparseMatrixCSC(obj.stencil.L1y)
    L2x = CuSparseMatrixCSC(obj.stencil.L2x)
    L2y = CuSparseMatrixCSC(obj.stencil.L2y)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)*maximum(densityInv))

    flux = zeros(size(psi))

    prog = Progress(nEnergies-1,1)

    uOUnc = zeros(nx*ny);
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n])#.*sqrt.(obj.gamma); # TODO: check sigma hat to be divided by sqrt(gamma)

        # set boundary condition
        if obj.settings.problem != "validation" # validation testcase sets beam in initial condition
            for k = 1:nq
                for j = 1:nx
                    psi[j,1,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[1],n-1);
                    psi[j,end,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[j],obj.settings.yMid[end],n-1);
                end
                for j = 1:ny
                    psi[1,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[1],obj.settings.yMid[j],n-1);
                    psi[end,j,k] = PsiBeam(obj,obj.qReduced[k,:],energy[n-1],obj.settings.xMid[end],obj.settings.yMid[j],n-1);
                end
            end
        end

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                uOUnc[idx] = psi[i,j,:]'*obj.MReduced[1,:];
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:] .+ uOUnc) * obj.csd.S[n-1] ./ obj.densityVec ;

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

            WAzW .= (Az*W)'*W # Az  = Az^T
            WAbsAzW .= (AbsAz*W)'*W
            WAbsAxW .= (AbsAx*W)'*W
            WAxW .= (Ax*W)'*W # Ax  = Ax^T

            K .= K .- dE*(L2x*K*WAxW + L2y*K*WAzW + L1x*K*WAbsAxW + L1y*K*WAbsAzW);

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L = W*S';

            XL2xX .= X'*L2x*X
            XL2yX .= X'*L2y*X
            XL1xX .= X'*L1x*X
            XL1yX .= X'*L1y*X

            L .= L .- dE*(Ax*L*XL2xX' + Az*L*XL2yX' + AbsAx*L*XL1xX' + AbsAz*L*XL1yX');
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            # impose boundary condition
            #X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*L2x*X
            XL2yX .= X'*L2y*X
            XL1xX .= X'*L1x*X
            XL1yX .= X'*L1y*X

            WAzW .= W'*Az*W
            WAbsAzW .= W'*AbsAz*W
            WAbsAxW .= W'*AbsAx*W
            WAxW .= W'*Ax*W

            S .= S .- dE.*(XL2xX*S*WAxW + XL2yX*S*WAzW + XL1xX*S*WAbsAxW + XL1yX*S*WAbsAzW);
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
        K .= K .+ dE * Mat2Vec(psi) * (obj.MReduced' * (Diagonal(Dvec) * W) );
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        XNew,_,_ = svd!(K);

        MUp .= XNew' * X;

        ################## L-step ##################
        L = W*S';
        L = L .+dE*Diagonal(Dvec)*obj.MReduced*(Mat2Vec(psi)'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*Mat2Vec(psi))*obj.MReduced'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        for i = 1:nx
            for j = 1:ny
                idx = (i-1)*ny + j
                uOUnc[idx] = psi[i,j,:]'*obj.MReduced[1,:];
            end
        end
        obj.dose .+= 0.5*dE * (X*S*W[1,:]+uOUnc) * obj.csd.S[n] ./ obj.densityVec;
        
        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(S);
    # return solution and dose
    return X*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*W*V,W*V,obj.dose,psi;

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

        D = Diagonal((sigmaS[1] .- Dvec));#.*ET);

        # perform time update
        uTilde = u .- dE * RhsOld(obj,u);

        for j = 1:size(uNew,1)
            uNew[j,:] = (Id .+ dE*D)\uTilde[j,:];
        end
        
        # update dose
        obj.dose .+= dE * uNew[:,1] * obj.csd.SMid[n-1] ./( 1 + (n==2||n==nEnergies));

        u .= uNew;

        next!(prog) # update progress bar
    end
    # return end time and solution
    return 0.5*sqrt(obj.gamma[1])*u,obj.dose./ obj.densityVec;

end

function CudaSolveFirstCollisionSourceDLR4thOrder(obj::SolverCSD{T}) where {T<:AbstractFloat}
    # Get rank
    r=obj.settings.r;

    eTrafo = T.(obj.csd.eTrafo);
    energy = T.(obj.csd.eGrid);
    S = obj.csd.S;

    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;

    x = obj.settings.xMid;
    y = obj.settings.yMid;

    sigmaO1Inv = 10000.0;
    sigmaO3Inv = 10000.0;
    pos_beam = [obj.settings.x0,obj.settings.y0];

    counterPNG = 0

    # Set up initial condition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-17;
    if obj.settings.problem == "LineSource" || obj.settings.problem == "2DHighD" || obj.settings.problem == "2DHighLowD" # determine relevant directions in IC
        psi = SetupIC(obj,obj.Q.pointsxyz);
        idxFullBeam = findall(psi .> floorPsiAll)
        idxBeam = findall(psi[idxFullBeam[1][1],idxFullBeam[1][2],:] .> floorPsi)
        psi = psi[:,:,idxBeam]
    else #if obj.settings.problem == "lung" || obj.settings.problem == "lungOrig" || obj.settings.problem == "liver" || obj.settings.problem == "validation" # determine relevant directions in beam
        psiBeam = zeros(nq)
        for k = 1:nq
            psiBeam[k] = PsiBeam(obj,T.(obj.Q.pointsxyz[k,:]),T(obj.settings.eMax),obj.settings.x0,obj.settings.y0,1)
        end
        idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
        psi = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    end
    
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:];
    obj.MReduced = obj.M[:,idxBeam];
    obj.OReduced = obj.O[idxBeam,:];
    obj.M = obj.M[:,idxBeam]
    nq = length(idxBeam);
    
    println("reduction of ordinates is ",(nq-length(idxBeam))/nq*100.0," percent")
    nq = length(idxBeam);

    # define density matrix
    densityInv = Diagonal(1.0 ./obj.density);
    # define identity matrix
    Id = Diagonal(ones(T,N));

    # Low-rank approx of init data:
    X,_,_ = svd!(zeros(T,nx*ny,r));
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
    WAzW = CUDA.zeros(T,r,r)

    XL2xX = CUDA.zeros(T,r,r)
    XL2yX = CUDA.zeros(T,r,r)

    MUp = CUDA.zeros(T,r,r)
    NUp = CUDA.zeros(T,r,r)

    L2x = CuSparseMatrixCSC(obj.stencil.L2x);
    L2y = CuSparseMatrixCSC(obj.stencil.L2y);

    Ax = CuSparseMatrixCSC(obj.pn.Ax)
    Az = CuSparseMatrixCSC(obj.pn.Az)

    XNew = CUDA.zeros(T,nx*ny,r)

    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;

    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE;

    println("CFL = ",dE/min(obj.settings.dx,obj.settings.dy)*maximum(densityInv))

    psiCPU = Ten2Vec(psi)
    psi = CuArray(psiCPU);
    M1 = CuArray(obj.M[1,:])
    M = CuArray(obj.M)
    sPow = obj.csd.S
    densityVec = CuArray(obj.densityVec);
    dose = CuArray(obj.dose);
    DvecCPU = zeros(T,obj.pn.nTotalEntries)

    prog = Progress(nEnergies-1,1);

    uOUnc = zeros(nx*ny);

    XX = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
    YY = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

    dE12 = Float32(0.5*dE);
    e1 = zeros(T,N); e1[1] = 1.0; e1 = CuArray(e1);

    intSigma = dE * SigmaAtEnergy(obj.csd,energy[1])[1];
    
    #loop over energy
    for n=2:nEnergies
        # compute scattering coefficients at current energy
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]);

        ############## Dose Computation ##############
        dose .+= dE12 * (X*S*(W' * e1)+ psi * M1) * sPow[n-1] ./ densityVec ;

        intSigma += dE * sigmaS[1];

        for q = 1:nq
            beamOmega = 10^5 * exp(-sigmaO1Inv*(obj.settings.Omega1-obj.qReduced[q,1])^2) * exp(-sigmaO3Inv*(obj.settings.Omega3-obj.qReduced[q,3])^2) * exp(-intSigma);
            for j = 1:ny
                beamy = normpdf(y[j] - eTrafo[n]*obj.qReduced[q,3],pos_beam[2],.1)
                if beamy < 1e-6 continue; end
                for i = 1:nx
                    beamx = normpdf(x[i] - eTrafo[n]*obj.qReduced[q,1],pos_beam[1],.1)
                    if beamx < 1e-6 continue; end
                    idx = vectorIndex(ny,i,j)
                    psiCPU[idx,q] = T(beamOmega * beamx * beamy )             
                end
            end
        end

        psi .= CuArray(psiCPU)

        
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                DvecCPU[i+1] = sigmaS[l+1]
            end
        end
        Dvec = CuArray(DvecCPU)

        if n > 2 # perform streaming update after first collision (before solution is zero)
            ################## K-step ##################
            K .= X*S;

            WAxW .= (Ax*W)'*W # Ax  = Ax^T
            WAzW .= (Az*W)'*W # Az  = Az^T

            k1 .= -L2x*K*WAxW .- L2y*K*WAzW;
            K .= K .+ dE .* k1 ./ 6;
            k1 .= -L2x*(K.+dE12.*k1)*WAxW .- L2y*(K.+dE12.*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(K.+dE12.*k1)*WAxW .- L2y*(K.+dE12.*k1)*WAzW;
            K .+= 2 * dE .* k1 ./ 6;
            k1 .= -L2x*(K.+dE.*k1)*WAxW .- L2y*(K.+dE.*k1)*WAzW;
            K .+= dE .* k1 ./ 6;

            XNew,_,_ = svd!(K);

            MUp .= XNew' * X;
            ################## L-step ##################
            L .= W*S';

            XL2xX .= X'*(L2x*X)
            XL2yX .= X'*(L2y*X)

            l1 .= -Ax*L*XL2xX' .- Az*L*XL2yX';
            L .= L .+ dE .* l1 ./ 6;
            l1 .= -Ax*(L+dE12*l1)*XL2xX' .- Az*(L+dE12*l1)*XL2yX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -Ax*(L+dE12*l1)*XL2xX' .- Az*(L+dE12*l1)*XL2yX';
            L .+= 2 * dE .* l1 ./ 6;
            l1 .= -Ax*(L+dE*l1)*XL2xX' .- Az*(L+dE*l1)*XL2yX';
            L .+= dE .* l1 ./ 6;
                    
            WNew,_,_ = svd!(L);

            NUp .= WNew' * W;
            W .= WNew;
            X .= XNew;

            ################## S-step ##################
            S .= MUp*S*(NUp')

            XL2xX .= X'*(L2x*X)
            XL2yX .= X'*(L2y*X)

            WAxW .= W'*(Ax*W)
            WAzW .= W'*(Az*W)

            s1 = -XL2xX*S*WAxW .- XL2yX*S*WAzW;
            s2 = -XL2xX*(S+dE12*s1)*WAxW .- XL2yX*(S+dE12*s1)*WAzW;
            s3 = -XL2xX*(S+dE12*s2)*WAxW .- XL2yX*(S+dE12*s2)*WAzW;
            s4 = -XL2xX*(S+dE*s3)*WAxW .- XL2yX*(S+dE*s3)*WAzW;

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
        L .= W*S';
        L .= L .+dE*Diagonal(Dvec)*M*(psi'*X);

        WNew,_,_ = svd!(L);

        NUp .= WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        S .= S .+dE*(X'*psi)*M'*(Diagonal(Dvec)*W);

        ############## Dose Computation ##############
        dose .+= dE12 * (X*S*(W' * e1)+psi * M1) * sPow[n] ./ densityVec;

        ######## Plotting #########
        if mod(n-1,Int(round(7*nEnergies/474/7))) == 0 && false
            counterPNG += 1;
            
            U,Sigma,V = svd(S);
            close("all")
            fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
            ax = gca()
            pcolormesh(YY,XX,Vec2Mat(s.NCellsX,s.NCellsY,X*S*W[1,:]+psi * M1)[2:end-1,2:end-1]')
            ax.tick_params("both",labelsize=20) 
            plt.xlabel("x", fontsize=20)
            plt.ylabel("y", fontsize=20)
            plt.title("scalar flux,  E = $(round(obj.csd.eGrid[n], digits=3)) MeV", fontsize=25)
            tight_layout()
            if counterPNG < 10
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_000$(counterPNG).png")
            elseif counterPNG < 100
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_00$(counterPNG).png")
            elseif counterPNG < 1000
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_0$(counterPNG).png")
            else
                savefig("output/gifPhi/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_$(counterPNG).png")
            end

            close("all")
            fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
            ax = gca()
            pcolormesh(YY,XX,Vec2Mat(s.NCellsX,s.NCellsY,X*U[:,1])[2:end-1,2:end-1]')
            ax.tick_params("both",labelsize=20) 
            #plt.xlabel("x", fontsize=20)
            #plt.ylabel("y", fontsize=20)
            plt.title("dominant spatial mode,  E = $(round(obj.csd.eGrid[n], digits=3)) MeV", fontsize=25)
            tight_layout()
            if counterPNG < 10
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_000$(counterPNG).png")
            elseif counterPNG < 100
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_00$(counterPNG).png")
            elseif counterPNG < 1000
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_0$(counterPNG).png")
            else
                savefig("output/gifX/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)_$(counterPNG).png")
            end
            
            # write modal basis
            writedlm("output/gifW/W_$(counterPNG)",W*V)
            #break
        end

        next!(prog) # update progress bar
    end

    U,Sigma,V = svd!(Matrix(S));
    # return solution and dose
    return Matrix(X)*U, 0.5*sqrt(obj.gamma[1])*Sigma, obj.O*Matrix(W)*V,Matrix(W)*V,Vector(dose),Matrix(psi);

end