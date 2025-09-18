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
using Interpolations
using TimerOutputs
using Random, Distributions


include("testCase_CSD/CSD.jl")
include("testCase_CSD/PNSystem.jl")
include("quadratures/Quadrature.jl")
include("testCase_CSD/utils.jl")
include("testCase_CSD/stencils.jl")

mutable struct solverCSD{T<:AbstractFloat}
    # spatial grid of cell interfaces
    x::Array{T};
    y::Array{T};
    z::Array{T};

    order::Int;
    
    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{T,1};

    # functionalities of the CSD approximation
    csd::CSD;

    # functionalities of the PN system
    pn::PNSystem;

    # stencil matrices
    stencil::UpwindStencil3DCUDA;

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

    #parameters for uq 
    uncertParam::String #type of uncertainty -> "1" - spatial, "2" - range
    alpha::Array{Float64,1}; #either positional shift in x,y-dir (2D) or scaling of range (1D)

    OReduced::Array{T,2};
    MReduced::Array{T,2};
    qReduced::Array{T,2};
 
    # constructor
    function solverCSD(settings)
        x = settings.x;
        y = settings.y;
        z = settings.z;

        nx = settings.NCellsX;
        ny = settings.NCellsY;
        nz = settings.NCellsZ;

        order = settings.order

        # setup flux matrix
        gamma = zeros(T,settings.nPN+1);
        for i = 1:settings.nPN+1
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        # construct CSD fields
        csd = CSD(settings,T);

        # set density vector
        density = T.(settings.density);

        # allocate dose vector
        dose = zeros(T,nx*ny*nz)
        pn = PNSystem(settings,T)
        SetupSystemMatricesSparse(pn)
        stencil = UpwindStencil3DCUDA(settings,order);
  
        Norder = (settings.nPN+1)^2
            # collect boundary indices
            if order == 1
                boundaryIdx = zeros(Int,2*nx*ny+2*ny*nz + 2*nx*nz)
                Threads.@threads for i = 1:nx
                    for k = 1:nz
                        j = 1;
                        boundaryIdx[(i-1)*nz*2+(k-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                        j = ny;
                        boundaryIdx[(i-1)*nz*2+(k-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                    end
                end
                Threads.@threads for i = 1:nx
                    for j = 1:ny
                        k = 1;
                        boundaryIdx[2*nx*nz+2*(i-1)*ny+(j-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                        k = nz;
                        boundaryIdx[2*nx*nz+2*(i-1)*ny+(j-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                    end
                end
        
                Threads.@threads for j = 1:ny
                    for k = 1:nz
                        i = 1;
                        boundaryIdx[2*nx*ny+2*nx*nz+2*(j-1)*nz+(k-1)*2+1] = vectorIndex(nx,ny,i,j,k)
                        i = nx;
                        boundaryIdx[2*nx*ny+2*nx*nz+2*(j-1)*nz+(k-1)*2+2] = vectorIndex(nx,ny,i,j,k)
                    end
                end
            elseif order == 2
                boundaryIdx = zeros(Int,4*nx*ny+4*ny*nz+4*nx*nz)
                counter = 0;
                Threads.@threads for i = 1:nx
                    Threads.@threads for k = 1:nz
                        j = 1;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+1] = vectorIndex(nx,ny,i,j,k)
                        j = 2;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+2] = vectorIndex(nx,ny,i,j,k)
                        j = ny;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+3] = vectorIndex(nx,ny,i,j,k)
                        j = ny-1;
                        boundaryIdx[(i-1)*nz*4+(k-1)*4+4] = vectorIndex(nx,ny,i,j,k)
                    end
                end
                Threads.@threads for i = 1:nx
                    Threads.@threads for j = 1:ny
                        k = 1;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+1] = vectorIndex(nx,ny,i,j,k)
                        k = 2;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+2] = vectorIndex(nx,ny,i,j,k)
                        k = nz;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+3] = vectorIndex(nx,ny,i,j,k)
                        k = nz - 1;
                        boundaryIdx[4*nx*nz+(i-1)*ny*4+(j-1)*4+4] = vectorIndex(nx,ny,i,j,k)
                    end
                end
                Threads.@threads for j = 1:ny
                    Threads.@threads for k = 1:nz
                        i = 1;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+1] = vectorIndex(nx,ny,i,j,k)
                        i = 2;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+2] = vectorIndex(nx,ny,i,j,k);
                        i = nx;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+3] = vectorIndex(nx,ny,i,j,k)
                        i = nx - 1;
                        boundaryIdx[4*nx*ny+4*nx*nz+(j-1)*nz*4+(k-1)*4+4] = vectorIndex(nx,ny,i,j,k)
                    end
                end
            end
            # setup quadrature
            qorder = 1; 
            if iseven(qorder) qorder += 1; end # make quadrature odd to ensure direction (0,1,0) is contained
            qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
            Q = Quadrature(qorder,qtype);

            Norder = pn.nTotalEntries;
            O,M = ComputeTrafoMatrices(Q,Norder,settings.nPN);

        densityVec = Ten2Vec(density);
        uncertParam = "1"
        alpha = [0.0,0.0]

        new{T}(T.(x),T.(y),T.(z),order,settings,gamma,csd,pn,stencil,density,densityVec,dose,boundaryIdx,Q,T.(O),T.(M),T,uncertParam,alpha);
    end
end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""

function SetupIC(obj::solverCSD{T},pointsxyz::Matrix{Float64}) where {T<:AbstractFloat}
    nq = size(pointsxyz)[1];
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    psi = zeros(T,obj.settings.NCellsX,obj.settings.NCellsY,obj.settings.NCellsZ,nq);

    for i = 2:nx-3
        for j = 2:ny-3
            for k = 2:nz-3
                for q = 1:nq 
                    psi[i,j,k,q] = PsiBeam(obj,T.(pointsxyz[q,:]),T(obj.settings.eMax),[obj.settings.xMid[i]],[obj.settings.yMid[j]],[obj.settings.zMid[k]],1)
                end
            end
        end
    end
    
    return psi;
end

function PsiBeam(obj::solverCSD{T},Omega::Array{T,1},E::T,x::Array{Float64,1},y::Array{Float64,1},z::Array{Float64,1},n::Int) where {T<:AbstractFloat}
    E0 = obj.settings.eMax;
    sigmaEInv = 1000.0;
    nB = size(obj.settings.Omega1,1)
    beam = 0.0;
    for b=1:nB
        if obj.uncertParam == "1"
            pos_beam = [obj.settings.x0[b],obj.settings.y0[b],obj.settings.z0[b]] .+ rotate_bev_to_xyz([obj.settings.Omega1[b],obj.settings.Omega2[b],obj.settings.Omega3[b]])*[obj.alpha[1],obj.alpha[2],0];
        elseif obj.uncertParam == "2"
            if obj.settings.particle == "Protons"
                #This uses Bragg-Kleemann rule to translate energy, i.e. range uncertainty, to a shift in depth, alpha=0.0022 and p=1.77 may need to be changed according to density?
                pos_beam = [obj.settings.x0[b],obj.settings.y0[b],obj.settings.z0[b]]  .+ rotate_bev_to_xyz([obj.settings.Omega1[b],obj.settings.Omega2[b],obj.settings.Omega3[b]])*[0,0, sqrt((0.0022*1.77*(eKin^0.77))^2*(obj.settings.sigmaE*(obj.settings.eMax-obj.settings.eRest)))];
            else 
                println("Warning: Transfer from range to depth uncert. for electrons not implemented, using parameters for protons.")
                #This uses Bragg-Kleemann rule to translate energy, i.e. range uncertainty, to a shift in depth, alpha=0.0022 and p=1.77 need to be changed for electrons and maybe according to density?
                pos_beam = [obj.settings.x0[b],obj.settings.y0[b],obj.settings.z0[b]]  .+ rotate_bev_to_xyz([obj.settings.Omega1[b],obj.settings.Omega2[b],obj.settings.Omega3[b]])*[0,0, sqrt((0.0022*1.77*(eKin^0.77))^2*(obj.settings.sigmaE*(obj.settings.eMax-obj.settings.eRest)))];
            end
        else 
            pos_beam = [obj.settings.x0[b],obj.settings.y0[b],obj.settings.z0[b]]
        end
        space_beam = normpdf(x,pos_beam[1],obj.settings.sigmaX).*normpdf(y,pos_beam[2],obj.settings.sigmaY).*normpdf(z,pos_beam[3],obj.settings.sigmaZ);
        omega_beam = normpdf(Omega[1],obj.settings.Omega1[b],0.01)*normpdf(Omega[2],obj.settings.Omega2[b],0.01)*normpdf(Omega[3],obj.settings.Omega3[b],0.01);
        beam += prod(space_beam) * omega_beam
    end
    return beam#.* normpdf(E,obj.settings.eMax,obj.settings.sigmaE) #.*obj.csd.S[n+1].* normpdf(E,obj.settings.eMax,obj.settings.sigmaE)
end

function solve_rankAdaptive(obj::solverCSD{T},sample::Array{S,1}) where {T,S<:AbstractFloat}
    obj.alpha = sample;
    model = obj.settings.model
    g = solve_rankAdaptive(obj, model)
    return g
end

function solve_rankAdaptive(obj::solverCSD{T}, model::String="Boltzmann") where {T<:AbstractFloat}
    # Get rank
    r=Int(floor(obj.settings.r / 2));
    order = obj.order
 
    eTrafo = obj.csd.eTrafo;
    energy = obj.csd.eGrid;
    #S = obj.csd.S;
 
    nx = obj.settings.NCellsX;
    ny = obj.settings.NCellsY;
    nz = obj.settings.NCellsZ;
    nq = obj.Q.nquadpoints;
    N = obj.pn.nTotalEntries;
    s = obj.settings;


    # Set up initial condition and store as matrix
    floorPsiAll = 1e-1;
    floorPsi = 1e-10;
    psiBeam = zeros(nq)
    for k = 1:nq
        psiBeam[k] = PsiBeam(obj,T.(obj.Q.pointsxyz[k,:]),T.(obj.settings.eMax),[obj.settings.x0],[obj.settings.y0],[obj.settings.z0],1)
    end
    idxBeam = findall( psiBeam .> floorPsi*maximum(psiBeam) );
    psiCPU = SetupIC(obj,obj.Q.pointsxyz[idxBeam,:]);
    obj.qReduced = obj.Q.pointsxyz[idxBeam,:]
    #obj.M = obj.M[:,idxBeam]
    MReduced = CuArray(T.(obj.M[:,idxBeam]))
    M1 = MReduced[1,:]
    obj.OReduced = obj.O[idxBeam,:]
    nq = length(idxBeam);
    e1 = zeros(T,N); e1[1] = 1.0; e1 = CuArray(e1);    
    # Low-rank approx of init data:
    X,_,_ = svd(zeros(T,nx*ny*nz,r));
    W,_,_ = svd(zeros(T,N,r));
 
    # rank-r truncation:
    X = CuArray(X[:,1:r]);
    W = CuArray(W[:,1:r]);
    S = CUDA.zeros(T,r,r);
    K = CUDA.zeros(T,size(X));
 
    MUp = CUDA.zeros(T,r,r)
    NUp = CUDA.zeros(T,r,r)
 
    # impose boundary condition
    X[obj.boundaryIdx,:] .= 0.0;
 
    nEnergies = length(eTrafo);
    dE = eTrafo[2]-eTrafo[1];
    obj.settings.dE = dE
    densityVec = CuArray(obj.densityVec) 
    Id = Diagonal(ones(T,N));
    idx = Base.unique(i -> obj.densityVec[i], 1:length(obj.densityVec))
    idxK = Vector{Vector{Int64}}([])
    el_time = @elapsed begin
    for k=1:length(idx)
    push!(idxK,findall(i->(i==obj.densityVec[idx[k]]),obj.densityVec))
    end
    end
    println("Time for material indexing = $el_time")
    rVec = r .* ones(2,nEnergies)
    t = 0;
 
    counterPNG = 0;
 
    dose = CuArray(obj.dose);
    flux = zeros(T,size(psiCPU))

    stencil = UpwindStencil3DCUDA(obj.settings, order)
    D⁺₁ = stencil.D⁺₁
    D⁺₂ = stencil.D⁺₂
    D⁺₃ = stencil.D⁺₃
    D⁻₁ = stencil.D⁻₁
    D⁻₂ = stencil.D⁻₂
    D⁻₃ = stencil.D⁻₃

    CUDA.reclaim()


    Ax = CuArray(obj.pn.Ax)
    Ay = CuArray(obj.pn.Ay)
    Az = CuArray(obj.pn.Az)

    Σ₁, T₁ = eigen(Ax)

    Ax = nothing
    T₁⁻¹ = T₁'
    Σ₁⁺ =  max.(Σ₁,0);
    Σ₁⁻ =  min.(Σ₁,0);
    Σ₁ = nothing
     
    Σ₂, T₂ = eigen(Ay)

    Ay = nothing

    T₂⁻¹ = T₂'
    Σ₂⁺ = max.(Σ₂, 0);
    Σ₂⁻ = min.(Σ₂, 0);
    Σ₂ = nothing

    Σ₃, T₃  = eigen(Az)
    Az = nothing
    
    T₃⁻¹ = T₃'
    Σ₃⁺ = max.(Σ₃,0);
    Σ₃⁻ = min.(Σ₃,0);
    Σ₃ = nothing

    ∫Y₀⁰dΩ = T(4 * pi / sqrt(4 * pi)); 
    Sinv_CPU = zeros(T,nx*ny*nz)
    S_tmp = zeros(T,nx*ny*nz)
    S_CPU = zeros(T,nx,ny,nz)
    wMat = T.(CuArray(matComp(obj.settings.densityHU[:]).*obj.settings.density[:]'./100));
    Nmat = size(wMat,1)
    prog = Progress(nEnergies-1,1)
    CUDA.reclaim()
    println("Starting energy loop")
    for n=2:nEnergies
        dE = energy[n-1] - energy[n]
        dEGrid = energy[n-1] - energy[n]
        # compute scattering coefficients at current energy
    if obj.settings.waterEq 
        sigmaS = SigmaAtEnergy(obj.csd,energy[n]) 
    else
        sigmaS = SigmaAtEnergyandX(obj.csd,energy[n])
    end

        DvecCPU = zeros(obj.pn.nTotalEntries,Nmat)
    for j=1:Nmat
        for l = 0:obj.pn.N
            for k=-l:l
                i = GlobalIndex( l, k );
                DvecCPU[i+1,j] = sigmaS[l+1,j] #Boltzmann
            end
        end
    end
        sigmaS1 = CuArray(T.(sigmaS[1,:]))
        Dvec = CuArray(T.(DvecCPU))

        for j=1:length(idx)
        Sinv_CPU[idxK[j]] .= 1 ./obj.csd.S[n,j]
        S_tmp[idxK[j]] .= obj.csd.S[n,j]
        end
        S_t = CuArray(S_tmp)
        S_CPU = reshape(S_tmp,nx,ny,nz)
        Sinv = CuArray(Sinv_CPU)

        solveFlux_rev_3D!(obj,psiCPU./S_CPU, flux)
        psiCPU .-= dE .* flux  
        psi = CuArray(Ten2Vec(psiCPU))
        sigma_w = sigmaS1' * wMat
        psi ./= (1 .+ dE .* Sinv.*sigma_w')
        psi[obj.boundaryIdx,:] .= 0
        psiCPU = reshape(Matrix(psi),nx,ny,nz,nq)

        if n > 2 # perform streaming update after first collision (before solution is zero)
        function FLx(t, L,idx)
            return - (Σ₁⁺.*L*(X'*D⁺₁*(Sinv.*X))' .+ Σ₁⁻.*L*(X'*D⁻₁*(Sinv.*X))')
        end
    
        function FLy(t, L,idx)
            return - (Σ₂⁺.*L*(X'*D⁺₂*(Sinv.*X))' .+ Σ₂⁻.*L*(X'*D⁻₂*(Sinv.*X))')
        end
                                                                                                                                                        
        function FLz(t, L,idx)
            return - (Σ₃⁺.*L*(X'*D⁺₃*(Sinv.*X))' .+ Σ₃⁻.*L*(X'*D⁻₃*(Sinv.*X))')
        end

        function FKx(t, K,idx)
            return - (D⁺₁*(Sinv.*K)*((W'*T₁)*(Σ₁⁺.*(T₁⁻¹*W))) .+ D⁻₁*(Sinv.*K)*((W'*T₁)*(Σ₁⁻.*(T₁⁻¹*W))))
        end
    
        function FKy(t, K,idx)
            return - (D⁺₂*(Sinv.*K)*((W'*T₂)*(Σ₂⁺.*(T₂⁻¹*W))) .+ D⁻₂*(Sinv.*K)*((W'*T₂)*(Σ₂⁻.*(T₂⁻¹*W))))

        end

        function FKz(t, K,idx)
            return - (D⁺₃*(Sinv.*K)*((W'*T₃)*(Σ₃⁺.*(T₃⁻¹*W))) .+ D⁻₃*(Sinv.*K)*((W'*T₃)*(Σ₃⁻.*(T₃⁻¹*W))))
        end

            ################## K-step ##################
            X[obj.boundaryIdx,:] .= 0.0;

            K = X*S;
            K .= rk4_idx(dE, FKx, K,order)
            K .= rk4_idx(dE, FKy, K,order)
            K .= rk4_idx(dE, FKz, K,order)
            Xtmp,_,_ = svd([X K]); MUp = Xtmp' * X;
        
            ################## L-step ##################
            L = T₁⁻¹*W*S';
            L .= T₁*rk4_idx(dE, FLx, L,order)
            L .= T₂⁻¹*L
            L .= T₂*rk4_idx(dE, FLy, L,order)
            L .= T₃⁻¹*L
            L .= T₃*rk4_idx(dE, FLz, L,order)

            Wtmp,_,_ = svd([W L]); NUp = Wtmp'*W;
            X = Xtmp;
            W = Wtmp;

            # impose boundary condition
            X[obj.boundaryIdx,:] .= 0.0;
            ################## S-step ##################
            function FSx(t, S,idx)
            return - ((X'*D⁺₁*(Sinv.*X))*S*((W'*T₁)*(Σ₁⁺.*(T₁⁻¹*W))) .+ (X'*D⁻₁*(Sinv.*X))*S*((W'*T₁)*(Σ₁⁻.*(T₁⁻¹*W))))
        end
    
        function FSy(t, S,idx)
            return - ((X'*D⁺₂*(Sinv.*X))*S*((W'*T₂)*(Σ₂⁺.*(T₂⁻¹*W))) .+ (X'*D⁻₂*(Sinv.*X))*S*((W'*T₂)*(Σ₂⁻.*(T₂⁻¹*W))))
        end

        function FSz(t, S,idx)
            return - ((X'*D⁺₃*(Sinv.*X))*S*((W'*T₃)*(Σ₃⁺.*(T₃⁻¹*W))) .+ (X'*D⁻₃*(Sinv.*X))*S*((W'*T₃)*(Σ₃⁻.*(T₃⁻¹*W))))
        end
            S = MUp*S*(NUp')
            S .= rk4_idx(dE, FSx, S,order)
            S .= rk4_idx(dE, FSy, S,order)
            S .= rk4_idx(dE, FSz, S,order)

            # truncate
            X, S, W = truncateCUDA(obj,T.(X),T.(S),T.(W));
            r = size(S,1)
        end
    
        ############# Out Scattering ##############

        ################## L-step ##################
        L = W*S';
        L0 = L
        if model == "FP"
        for j=1:Nmat
            #L += dE*(Dvec[:,j].-sigmaS[1,j]).*(L0*X'*(wMat[j,:].*Sinv.*X));
            implicit_L_step!(L,T.(Dvec[:,j].-sigmaS[1,j]), X'*(wMat[j,:].*Sinv.*X), dE)
        end
    else
            for j=1:Nmat
            # L += dE*(Dvec[:,j].-sigmaS[1,j]).*(L0*X'*(wMat[j,:].*Sinv.*X));
            implicit_L_step!(L,T.(Dvec[:,j].-sigmaS[1,j]), X'*(wMat[j,:].*Sinv.*X), dE)
        end
    end
        W,S1,S2 = svd(L)
        S .= S2 * Diagonal(S1)
        ############## In Scattering ##############

        ################## K-step ##################
        X[obj.boundaryIdx,:] .= 0.0;
        K = X*S;
        for j=1:Nmat
        K += dE * wMat[j,:].*Sinv.* psi * MReduced'*(Dvec[:,j].*W); 
        end
        K[obj.boundaryIdx,:] .= 0.0; # update includes the boundary cell, which should not generate a source, since boundary is ghost cell. Therefore, set solution at boundary to zero

        Xtmp,_,_ = svd([X K]); MUp = Xtmp' * X;
        
        ################## L-step ##################
        L = W*S';
        for j=1:Nmat
        L += dE*Dvec[:,j].*MReduced*(X'*(wMat[j,:].*Sinv.*psi))';
        end
        Wtmp,_,_ = svd([W L]); NUp = Wtmp'*W;

        X = Xtmp;
        W = Wtmp;
        ################## S-step ##################
        S = MUp*S*(NUp')
        for j=1:Nmat
        S += dE*(X'*(wMat[j,:].*Sinv.*psi))*MReduced'*(Dvec[:,j].*W);
        end

        ############## Dose Computation ##############
        dose .+= dEGrid * (X*S*(W'*e1)+psi*M1) * ∫Y₀⁰dΩ#add density to compute dose instead of energy dep.
        # truncate
        X, S, W = truncateCUDA(obj,T.(X),T.(S),T.(W));
        next!(prog) # update progress bar
    end
    #normalize
    dose = dose./maximum(dose)
    return Vector(dose)
 end

function rk4_idx(Δt, f, u,t=0,order=4)
    if order ==1 
        k1 = Δt * f(t, u,1) 
        return u .+= k1
    else
        k1 = Δt * f(t, u,1)
        k2 = Δt * f(t + Δt/2, u .+ k1/2,2)
        k3 = Δt * f(t + Δt/2, u .+ k2/2,2)
        k4 = Δt * f(t + Δt, u .+ k3,3)
        return u .+= (k1 + 2*k2 + 2*k3 + k4) / 6
    end
end

function rk4(Δt, f, u, t=0,order=4)
    if order ==1 
        k1 = Δt * f(t, u) 
        u .+= k1 
    else
        k1 = Δt * f(t, u)
        k2 = Δt * f(t + Δt/2, u .+ k1/2)
        k3 = Δt * f(t + Δt/2, u .+ k2/2)
        k4 = Δt * f(t + Δt, u .+ k3)
        
        u .+= (k1 .+ 2 .*k2 .+ 2 .*k3 .+ k4) ./ 6
    end
    return u
end

function truncateCUDA(obj::solverCSD{T},X::CuArray{T,2},S::CuArray{T,2},W::CuArray{T,2}) where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:
    U,D,V = safe_svd(Matrix(S));
    rmax = -1;
    rMaxTotal = obj.settings.rMax;
    rMinTotal = 2;

    tmp = 0.0;
    tol = obj.settings.epsAdapt*norm(D)^obj.settings.adaptIndex;
    rmax = Int(floor(size(D,1)/2));

    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if tmp < tol
            rmax = j;
            break;
        end
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        println("Using rMax")
        rmax = rMaxTotal;
    end

    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,rMinTotal);
    Utilde = CuArray(U[:, 1:rmax])
    Vtilde = CuArray(V[:, 1:rmax])
    
    # return rank
    return X*Utilde, CuArray(diagm(D[1:rmax])), W*Vtilde;
end

function truncateToFixedRankCUDA(obj::solverCSD{T},X::CuArray{T,2},S::CuArray{T,2},W::CuArray{T,2}) where {T<:AbstractFloat}
    # Compute singular values of S and decide how to truncate:
    U,D,V = svd(Matrix(S));
    rmax = obj.settings.rMax;
    Utilde = CuArray(U[:, 1:rmax])
    Vtilde = CuArray(V[:, 1:rmax])

    # return rank
    return X*Utilde, CuArray(diagm(D[1:rmax])), W*Vtilde;
end

function solveFlux_rev_3D!(obj::solverCSD{T}, phi::Array{T,4}, flux::Array{T,4}) where {T<:AbstractFloat}
    # computes the numerical flux over cell boundaries for each ordinate
    idxPosPosPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxPosPosNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxPosNegPos = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].<0.0)  .&(obj.qReduced[:,3].>=0.0))
    idxPosNegNeg = findall((obj.qReduced[:,1].>=0.0) .&(obj.qReduced[:,2].<0.0)  .&(obj.qReduced[:,3].<0.0))
    idxNegPosPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].>=0.0))
    idxNegPosNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].>=0.0) .&(obj.qReduced[:,3].<0.0))
    idxNegNegPos = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].<0.0)  .&(obj.qReduced[:,3].>=0.0))
    idxNegNegNeg = findall((obj.qReduced[:,1].<0.0)  .&(obj.qReduced[:,2].<0.0)  .&(obj.qReduced[:,3].<0.0))

    if obj.order == 1 
        nx = collect(2:(obj.settings.NCellsX-1))
        ny = collect(2:(obj.settings.NCellsY-1))
        nz = collect(2:(obj.settings.NCellsZ-1))

        # PosPosPos
        for k in nz, j in ny, i in nx, q in idxPosPosPos
            # X direction
            s2, s3 = phi[i-1,j,k,q], phi[i,j,k,q]
            eastflux = s3
            westflux = s2 
            
            # Y direction
            s2, s3 = phi[i,j-1,k,q], phi[i,j,k,q]
            northflux = s3 
            southflux = s2

            # Z direction
            s2, s3 = phi[i,j,k-1,q], phi[i,j,k,q]
            topflux = s3 
            bottomflux = s2 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # PosPosNeg
        for k in nz, j in ny, i in nx, q in idxPosPosNeg
            # X direction
            s2, s3 = phi[i-1,j,k,q], phi[i,j,k,q]
            eastflux = s3
            westflux = s2 

            # Y direction
            s2, s3 = phi[i,j-1,k,q], phi[i,j,k,q]
            northflux = s3 
            southflux = s2

            # Z direction
            s2, s3 = phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s2 
            bottomflux = s3 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # PosNegPos
        for k in nz, j in ny, i in nx, q in idxPosNegPos
            # X direction
            s2, s3 = phi[i-1,j,k,q], phi[i,j,k,q]
            eastflux = s3
            westflux = s2 

            # Y direction
            s2, s3 = phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s2 
            southflux = s3 

            # Z direction
            s2, s3 = phi[i,j,k-1,q], phi[i,j,k,q]
            topflux = s3 
            bottomflux = s2 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # PosNegNeg
        for k in nz, j in ny, i in nx, q in idxPosNegNeg
            # X direction
            s2, s3 = phi[i-1,j,k,q], phi[i,j,k,q]
            eastflux = s3
            westflux = s2 

            # Y direction
            s2, s3 = phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s2 
            southflux = s3 

            # Z direction
            s2, s3 = phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s2 
            bottomflux = s3 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegPosPos
        for k in nz, j in ny, i in nx, q in idxNegPosPos
            # X direction
            s2, s3 = phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s2
            westflux = s3
            
            # Y direction
            s2, s3 = phi[i,j-1,k,q], phi[i,j,k,q]
            northflux = s3 
            southflux = s2

            # Z direction
            s2, s3 = phi[i,j,k-1,q], phi[i,j,k,q]
            topflux = s3 
            bottomflux = s2 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegPosNeg
        for k in nz, j in ny, i in nx, q in idxNegPosNeg
            # X direction
            s2, s3 = phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s2
            westflux = s3

            # Y direction
            s2, s3 = phi[i,j-1,k,q], phi[i,j,k,q]
            northflux = s3 
            southflux = s2

            # Z direction
            s2, s3 = phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s2 
            bottomflux = s3 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegNegPos
        for k in nz, j in ny, i in nx, q in idxNegNegPos
            # X direction
            s2, s3 = phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s2
            westflux = s3

            # Y direction
            s2, s3 = phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s2 
            southflux = s3 

            # Z direction
            s2, s3 = phi[i,j,k-1,q], phi[i,j,k,q]
            topflux = s3 
            bottomflux = s2 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegNegNeg
        for k in nz, j in ny, i in nx, q in idxNegNegNeg
            # X direction
            s2, s3 = phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s2
            westflux = s3
            
            # Y direction
            s2, s3 = phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s2 
            southflux = s3 

            # Z direction
            s2, s3 = phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s2 
            bottomflux = s3 

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end
    elseif obj.order == 2
        nx = collect(3:(obj.settings.NCellsX-2))
        ny = collect(3:(obj.settings.NCellsY-2))
        nz = collect(3:(obj.settings.NCellsZ-2))

        # PosPosPos
        for k in nz, j in ny, i in nx, q in idxPosPosPos
            # X direction
            s1, s2, s3, s4 = phi[i-2,j,k,q], phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            westflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            
            # Y direction
            s1, s2, s3, s4 = phi[i,j-2,k,q], phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            southflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-2,q], phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            bottomflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # PosPosNeg
        for k in nz, j in ny, i in nx, q in idxPosPosNeg
            # X direction
            s1, s2, s3, s4 = phi[i-2,j,k,q], phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            westflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            
            # Y direction
            s1, s2, s3, s4 = phi[i,j-2,k,q], phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            southflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q], phi[i,j,k+2,q]
            topflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            bottomflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # PosNegPos
        for k in nz, j in ny, i in nx, q in idxPosNegPos
            # X direction
            s1, s2, s3, s4 = phi[i-2,j,k,q], phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            westflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            # Y direction
            s1, s2, s3, s4 = phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q], phi[i,j+2,k,q]
            northflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            southflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-2,q], phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            bottomflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # PosNegNeg
        for k in nz, j in ny, i in nx, q in idxPosNegNeg
            # X direction
            s1, s2, s3, s4 = phi[i-2,j,k,q], phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q]
            eastflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            westflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            # Y direction
            s1, s2, s3, s4 = phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q], phi[i,j+2,k,q]
            northflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            southflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q], phi[i,j,k+2,q]
            topflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            bottomflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegPosPos
        for k in nz, j in ny, i in nx, q in idxNegPosPos
            # X direction
            s1, s2, s3, s4 = phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q], phi[i+2,j,k,q]
            eastflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            westflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Y direction
            s1, s2, s3, s4 = phi[i,j-2,k,q], phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            southflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-2,q], phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            bottomflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegPosNeg
        for k in nz, j in ny, i in nx, q in idxNegPosNeg
            # X direction
            s1, s2, s3, s4 = phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q], phi[i+2,j,k,q]
            eastflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            westflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Y direction
            s1, s2, s3, s4 = phi[i,j-2,k,q], phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q]
            northflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            southflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q], phi[i,j,k+2,q]
            topflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            bottomflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegNegPos
        for k in nz, j in ny, i in nx, q in idxNegNegPos
            # X direction
            s1, s2, s3, s4 = phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q], phi[i+2,j,k,q]
            eastflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            westflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Y direction
            s1, s2, s3, s4 = phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q], phi[i,j+2,k,q]
            northflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            southflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-2,q], phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q]
            topflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)
            bottomflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end

        # NegNegNeg
        for k in nz, j in ny, i in nx, q in idxNegNegNeg
            # X direction
            s1, s2, s3, s4 = phi[i-1,j,k,q], phi[i,j,k,q], phi[i+1,j,k,q], phi[i+2,j,k,q]
            eastflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            westflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Y direction
            s1, s2, s3, s4 = phi[i,j-1,k,q], phi[i,j,k,q], phi[i,j+1,k,q], phi[i,j+2,k,q]
            northflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            southflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            # Z direction
            s1, s2, s3, s4 = phi[i,j,k-1,q], phi[i,j,k,q], phi[i,j,k+1,q], phi[i,j,k+2,q]
            topflux = s2 + 0.5 * slopefit_rev(s1, s2, s3)
            bottomflux = s3 + 0.5 * slopefit_rev(s2, s3, s4)

            flux[i,j,k,q] = obj.qReduced[q,1] / obj.settings.dx * (eastflux - westflux) +
                            obj.qReduced[q,2] / obj.settings.dy * (northflux - southflux) +
                            obj.qReduced[q,3] / obj.settings.dz * (topflux - bottomflux)
        end
    else
        println("Order ", obj.order, "not implemented, choose either 1st or 2nd order!")
    end
end

function solveFlux!(obj::solverCSD{T}, phi::Array{T,4}, flux::Array{T,4}) where {T<:AbstractFloat}
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
        eastflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        southflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .* (eastflux-westflux) +
        obj.qReduced[q,2]./obj.settings.dy .* (northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end
    #PosNeg
    for j=ny,i=nx,k=nz,q = idxPosNegPos

        s1 = phi[i-2,j,k,q]
        s2 = phi[i-1,j,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i+1,j,k,q]
        eastflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit_rev(s2,s3,s4)
        southflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegPos
    for j=ny,i=nx,k=nz,q = idxNegPosPos
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        southflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegNeg
    for j=ny,i=nx,k=nz,q = idxNegNegPos
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit_rev(s2,s3,s4)
        southflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-2,q]
        s2 = phi[i,j,k-1,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j,k+1,q]
        upflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # PosPos
    for j=ny,i=nx,k=nz, q = idxPosPosNeg
        s1 = phi[i-2,j,k,q]
        s2 = phi[i-1,j,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i+1,j,k,q]
        eastflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        southflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

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
        eastflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit_rev(s2,s3,s4)
        southflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegPos
    for j=ny,i=nx,k=nz,q = idxNegPosNeg
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-2,k,q]
        s2 = phi[i,j-1,k,q]
        s3 = phi[i,j,k,q]
        s4 = phi[i,j+1,k,q]
        northflux = s3+0.5 .*slopefit_rev(s2,s3,s4)
        southflux = s2+0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1]./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end

    # NegNeg
    for j=ny,i=nx,k=nz,q = idxNegNegNeg
        s1 = phi[i-1,j,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i+1,j,k,q]
        s4 = phi[i+2,j,k,q]
        eastflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        westflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j-1,k,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j+1,k,q]
        s4 = phi[i,j+2,k,q]
        northflux = s3-0.5 .* slopefit_rev(s2,s3,s4)
        southflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        s1 = phi[i,j,k-1,q]
        s2 = phi[i,j,k,q]
        s3 = phi[i,j,k+1,q]
        s4 = phi[i,j,k+2,q]
        upflux = s3-0.5 .*slopefit_rev(s2,s3,s4)
        downflux = s2-0.5 .*slopefit_rev(s1,s2,s3)

        flux[i,j,k,q] = obj.qReduced[q,1] ./obj.settings.dx .*(eastflux-westflux) +
        obj.qReduced[q,2] ./obj.settings.dy .*(northflux-southflux) + obj.qReduced[q,3]./obj.settings.dz .* (upflux-downflux)
    end
end

# Van Leer limiter function
function vanleer_limiter(r)
    return (r + abs(r)) / (1 + abs(r))
end

# Slope fitting function using Van Leer limiter
function slopefit_rev(um1, u, up1)
    r = (u - um1) / (up1 - u + 1e-6)  # Compute the ratio with a small epsilon to avoid division by zero
    phi = vanleer_limiter(r)
    return phi * (up1 - u)
end


@inline minmod(x::T, y::T) where {T<:AbstractFloat} = ifelse(x < 0, clamp(y, x, 0.0), clamp(y, 0.0, x))

@inline function slopefit(left::T, center::T, right::T) where {T<:AbstractFloat}
    tmp = minmod(0.5 * (right - left),2.0 * (center - left));
    return minmod(2.0 * (right - center),tmp);
end

function implicit_L_step!(L::CuMatrix{T},
                            d::CuVector{T},
                            B::CuMatrix{T},
                            dt::T) where {T<:AbstractFloat}

    m, r = size(L)
    @assert length(d) == m

    # Transpose B on device (r×r)
    Bt = transpose(B)

    # Copy d to CPU to find unique values (d is 1D vector, cheap to copy)
    d_cpu = Array(d)
    unique_d = Base.unique(d_cpu)
    nd = length(unique_d)

    # Prepare storage for LU factors on GPU
    # We'll store a tuple of (U, L) for each unique d_i
    d_to_factor = Dict{T, Tuple{CuMatrix{T}, CuMatrix{T}}}()

    for dj in unique_d
        # Create identity matrix on GPU
        I_r_gpu = CUDA.CuArray(Matrix{T}(I, r, r))


        # Construct A = I - dt * dj * Bt on GPU
        A = I_r_gpu .- dt * dj * Bt

        # LU factorization on GPU (uses cuSOLVER)
        F = lu(A)

        # Store the triangular factors (U, L) as UpperTriangular and UnitLowerTriangular
        U = UpperTriangular(F.U)
        Lfac = UnitLowerTriangular(F.L)

        # Upload factors to dict
        d_to_factor[dj] = (U, Lfac)
    end

    # Group row indices by d_i value on CPU side
    groups = Dict{T, Vector{Int}}()
    for i in 1:m
        push!(get!(groups, d_cpu[i], Int[]), i)
    end

    # Solve (I - dt * d_i * Bt)ᵀ * L[i, :] = L_old[i, :] for each group on GPU
    for (dj, idx) in groups
        U, Lfac = d_to_factor[dj]

        # Extract rows of L corresponding to this group (view)
        rows = @view L[idx, :]

        rows_t = permutedims(rows)          # r × n  (will be overwritten)

        # forward solve  L * X = rows_t   →  result stored back into rows_t
        CUDA.CUBLAS.trsm!('L','L','N','U', one(T), Lfac, rows_t)

        # backward solve U * X = rows_t   →  final X is now in rows_t
        CUDA.CUBLAS.trsm!('L','U','N','N', one(T), U, rows_t)

        # copy the transposed result back into the original rows
        transpose!(rows, rows_t)
    end

    return L
end

function gpu_slopefit(a::Float32, b::Float32)
    if a * b <= 0
        return 0.0f0
    else
        return sign(a) * min(abs(a), abs(b))
    end
end

function solveFlux_rev_3D_CUDA!(
    flux::CuArray{Float32,4},
    phi::CuArray{Float32,4},
    qReduced::CuArray{Float32,2},
    dx::Float32, dy::Float32, dz::Float32,
    order::Int
)
    Nx, Ny, Nz, Nq = size(phi)
    total_cells = Nx * Ny * Nz * Nq
    threads = 256
    blocks = cld(total_cells, threads)

    @cuda threads=threads blocks=blocks kernel_flux!(
        flux, phi, qReduced, dx, dy, dz, order,
        Nx, Ny, Nz, Nq
    )
end

function kernel_flux!(
    flux, phi, qReduced, dx, dy, dz, order,
    Nx, Ny, Nz, Nq
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = Nx * Ny * Nz * Nq
    if i > total
        return
    end

    q = mod(i - 1, Nq) + 1
    k = mod(div(i - 1, Nq), Nz) + 1
    j = mod(div(i - 1, Nz * Nq), Ny) + 1
    x = div(i - 1, Ny * Nz * Nq) + 1

    if x < 2 || x > Nx-1 || j < 2 || j > Ny-1 || k < 2 || k > Nz-1
        return
    end

    vx = qReduced[q, 1]
    vy = qReduced[q, 2]
    vz = qReduced[q, 3]

    # Determine upwind neighbors
    iL = x - (vx >= 0 ? 1 : 0)
    iR = x + (vx >= 0 ? 0 : 1)
    jL = j - (vy >= 0 ? 1 : 0)
    jR = j + (vy >= 0 ? 0 : 1)
    kL = k - (vz >= 0 ? 1 : 0)
    kR = k + (vz >= 0 ? 0 : 1)

    fx = 0.0f0
    fy = 0.0f0
    fz = 0.0f0

    if order == 1
        fx = (phi[iR,j,k,q] - phi[iL,j,k,q]) * vx / dx
        fy = (phi[x,jR,k,q] - phi[x,jL,k,q]) * vy / dy
        fz = (phi[x,j,kR,q] - phi[x,j,kL,q]) * vz / dz

    elseif order == 2
        # X-direction slopes
        δLx = phi[iL,j,k,q] - phi[iL-1,j,k,q]
        δRx = phi[iR+1,j,k,q] - phi[iR,j,k,q]
        slope_x = (vx >= 0) ? minmod_gpu(δLx, phi[iL+1,j,k,q] - phi[iL,j,k,q]) :
                              minmod_gpu(phi[iR,j,k,q] - phi[iR-1,j,k,q], δRx)
        fx = ((phi[iR,j,k,q] - phi[iL,j,k,q]) / dx - slope_x / dx) * vx

        # Y-direction slopes
        δLy = phi[x,jL,k,q] - phi[x,jL-1,k,q]
        δRy = phi[x,jR+1,k,q] - phi[x,jR,k,q]
        slope_y = (vy >= 0) ? minmod_gpu(δLy, phi[x,jL+1,k,q] - phi[x,jL,k,q]) :
                              minmod_gpu(phi[x,jR,k,q] - phi[x,jR-1,k,q], δRy)
        fy = ((phi[x,jR,k,q] - phi[x,jL,k,q]) / dy - slope_y / dy) * vy

        # Z-direction slopes
        δLz = phi[x,j,kL,q] - phi[x,j,kL-1,q]
        δRz = phi[x,j,kR+1,q] - phi[x,j,kR,q]
        slope_z = (vz >= 0) ? minmod_gpu(δLz, phi[x,j,kL+1,q] - phi[x,j,kL,q]) :
                              minmod_gpu(phi[x,j,kR,q] - phi[x,j,kR-1,q], δRz)
        fz = ((phi[x,j,kR,q] - phi[x,j,kL,q]) / dz - slope_z / dz) * vz
    end

    flux[x,j,k,q] = fx + fy + fz
end

function implicit_update!(L::CuMatrix{T}, d::T, b::CuVector{T}, dt::T) where {T<:AbstractFloat}
    n, m = size(L)
    threads = (16, 16)  # block size
    blocks = (cld(m, threads[1]), cld(n, threads[2]))

    @cuda threads=threads blocks=blocks kernel_implicit_update!(L, d, b, dt, n, m)
    return L
end

function kernel_implicit_update!(
    L, d, b, dt, n, m
)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i ≤ m && j ≤ n
        @inbounds L[j,i] /= (1 - dt * d * b[j])
    end
    return
end
