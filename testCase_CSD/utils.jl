using Interpolations
function sph_cc(mu,phi,l,m)
    # Complex conjugates of coefficients.
    y = 0;
    z = computePlmx(mu,lmax=l,norm=SphericalHarmonics.Unnormalized())
    ma = abs(m);
    ind = Int(0.5*(l^2+l)+ma+1);
    
    y = y + sqrt((2*l+1)/(4*pi).*factorial(big(l-ma))./factorial(big(l+ma))).*(-1).^max(m,0).*exp(1im*m*phi).*z[ind];
    return y;
end

function sph_cc(mu,phi,l,m,z)
    # Complex conjugates of coefficients.
    ma = abs(m);
    ind = Int(0.5*(l^2+l)+ma+1);
    
    y = sqrt((2*l+1)/(4*pi).*factorial(big(l-ma))./factorial(big(l+ma))).*(-1).^max(m,0).*exp(1im*m*phi).*z[ind];
    return y;
end

function real_sph(mu,phi,l,k)
    # Complex conjugates of coefficients.
    if k > 0
        return Float64((-1)^k/sqrt(2)*(sph_cc(mu,phi,l,k)+(-1)^k*sph_cc(mu,phi,l,-k)));
    elseif k < 0
        return Float64(-(-1)^k*1im/sqrt(2)*(sph_cc(mu,phi,l,-k)-(-1)^k*sph_cc(mu,phi,l,k)));
    else
        return Float64(sph_cc(mu,phi,l,k));
    end
end

function real_sph(mu,phi,l,k,z)
    # Complex conjugates of coefficients.
    if k > 0
        return Float64((-1)^k/sqrt(2)*(sph_cc(mu,phi,l,k,z)+(-1)^k*sph_cc(mu,phi,l,-k,z)));
    elseif k < 0
        return Float64(-(-1)^k*1im/sqrt(2)*(sph_cc(mu,phi,l,-k,z)-(-1)^k*sph_cc(mu,phi,l,k,z)));
    else
        return Float64(sph_cc(mu,phi,l,k,z));
    end
end

function normpdf(x,mu,sigma)
    return 1 ./(sigma.*sqrt(2*pi))*exp.(-(x.-mu).^ 2 ./ 2 ./(sigma.^2));
end

function expm1div(x)
    # Function (exp(x)-1)/x that is accurate for x close to zero.
    y = 1+x*.5+x.^2/6;
    if abs(x)>2e-4;
        y = (exp(x)-1)./x;
    end
    return 1.0;#y;
end


function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*ny + j]
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
            v[(i-1)*ny + j,:] = mat[i,j,:]
        end
    end
    return v;
end

function Ten2Vec(ten)
    nx = size(ten,1)
    ny = size(ten,2)
    nz = size(ten,3)
    m = size(ten,4)
    v = zeros(nx*ny,m*nz);
    for i = 1:nx
        for j = 1:ny
            for l = 1:nz
                for k = 1:m
                    v[(i-1)*ny + j,(l-1)*m .+ k] = ten[i,j,l,k]
                end
            end
        end
    end
    return v;
end

function vectorIndex(ny,i,j)
    return (i-1)*ny + j;
end

function vectorIndex(nx, ny, i, j, k)
    return (k-1) * nx * ny + (j-1) * nx + i
end

function Vec2Ten(nx,ny,nz,v::Array{T,1}) where {T<:AbstractFloat}
    m = zeros(T,nx,ny,nz);
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                m[i,j,k] = v[vectorIndex(nx,ny,i,j,k)]
            end
        end
    end
    return m;
end

function Vec2Ten(nx,ny,nz,v::Array{T,2}) where {T<:AbstractFloat}
    n = size(v,2);
    m = zeros(T,nx,ny,nz,n);
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                m[i,j,k,:] = v[vectorIndex(nx,ny,i,j,k),:]
            end
        end
    end
    return m;
end

function Ten2Vec(mat::Array{T,4}) where {T<:AbstractFloat}
    nx = size(mat,1)
    ny = size(mat,2)
    nz = size(mat,3)
    m = size(mat,4)
    v = zeros(T,nx*ny*nz,m);
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                v[vectorIndex(nx,ny,i,j,k),:] = mat[i,j,k,:]
            end
        end
    end
    return v;
end

function Ten2Vec(mat::Array{T,3}) where {T<:AbstractFloat}
    nx = size(mat,1)
    ny = size(mat,2)
    nz = size(mat,3)
    v = zeros(T,nx*ny*nz);
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                v[vectorIndex(nx,ny,i,j,k)] = mat[i,j,k]
            end
        end
    end
    return v;
end

function Ten2Mat(mat::Array{T,4}) where {T<:AbstractFloat}
    nx = size(mat,1)
    ny = size(mat,2)
    nz = size(mat,3)
    v = zeros(T,nx*ny*nz,size(mat,4));
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                v[vectorIndex(nx,ny,i,j,k),:] = mat[i,j,k,:]
            end
        end
    end
    return v;
end

function Ten2Mat(mat::Array{T,5}) where {T<:AbstractFloat}
    nx = size(mat,1)
    ny = size(mat,2)
    nz = size(mat,3)
    v = zeros(T,nx*ny*nz,size(mat,4),size(mat,5));
    for i = 1:nx
        for j = 1:ny
            for k = 1:nz
                v[vectorIndex(nx,ny,i,j,k),:,:] = mat[i,j,k,:,:]
            end
        end
    end
    return v;
end

function tensorizePointSet(x, y)
    #assume x and y are vectors, then
    #z is the matrix of points that contains x_i y_j for all combinations

    n = length(x)
    m = length(y)
    z = zeros(n*m,2);
   
    for i = 1:n
        for j = 1:m
            z[(i-1)*n+j,:]= collect((x[i], y[j]));
        end
    end
    return z;
end
    
function tensorizeWeightSet(x, y)
   # assume x and y are vectors, then
   # z is the matrix of weights that contains x_i*y_j for all combinations
    n = length(x)
    m = length(y)
    z = zeros(n*m,);
    
    for i = 1:length(x)
        for j = 1:length(y)
            z[(i-1)*n+j] = x[i]*y[j];
        end
    end
    return z;
end

function cartesian_to_spherical(x)
    r = norm(x)
    θ = acos(x[3] / r)  # polar angle (theta)
    ϕ = atan(x[2], x[1])  # azimuthal angle (phi)
    return r, θ, ϕ
end

function Polyfun_E(E,E_min,E_max,no_nod_E,scaling)
# Note: to get the normalization correct we have divided by dE and have a factor
#3 for the slope. This routine only to be used as multiplier for the source.
#It can not be used for integrals of the form fi_x_fj ...
#scaling=F:    no scaling
#scaling=T:    scaling

dE = E_max - E_min

fun_E = zeros(no_nod_E)

if (no_nod_E == 1) 
  fun_E = 1.0
  if scaling 
    fun_E = fun_E * (1.0 / dE)
  end

elseif no_nod_E == 2

  E_mid = E_min + dE / 2.0
  x = (2 / dE) * (E-E_mid)
  fun_E[1]= 1.0
  fun_E[2] = x
  if scaling
    fun_E[1] = fun_E[1] * (1.0 / dE)
    fun_E[2] = fun_E[2] * (3.0 / dE)
  end
elseif no_nod_E == 3
  E_mid = E_min + dE / 2.0_dp
  x = (2 / dE) * (E - E_mid)
  fun_E[1] = 1.0
  fun_E[2] = x
  fun_E[3] = (1.0 / 2.0) * (3.0 * x^2 - 1.0)
  if (scaling) then
    fun_E[1] = fun_E[1] * (1.0 / dE)
    fun_E[2] = fun_E[2] * (3.0/ dE)
    fun_E[3] = fun_E[3] * (5.0 / dE)
  end

else
  println("polynomial order for energy not in valid range")
end
return fun_E
end

function combination_vectors(v1, v2)
    c1 = reduce(vcat, [fill(v1[i], length(v2)) for i in 1:length(v1)])
    c2 = repeat(v2, length(v1))
    return(hcat(c1, c2))
end

function SOBP(R1,xi,n,alpha=0.0022,p0=1.77)
    # sets up spread out bragg peak according to [Bortfeld, Schlegel 1996]
    # gives back ranges, energies and weights for the n beams so that peak spreads from depth R0=(1-xi)R1 to R1
    # alpha and p are parameters of Bragg-Kleemann rule, default according to [Bortfeld 1997] 
    # correction is done using p values according to [Jette, Chen 2011]
    correction_factors = [1.48 1.45 1.43 1.43 1.42 1.41;
                          1.46 1.43 1.42 1.41 1.40 1.38;
                          1.43 1.40 1.39 1.37 1.36 1.35;
                          1.40 1.37 1.34 1.33 1.32 1.30;
                          1.34 1.32 1.29 1.27 1.26 1.24];
    E_tab = [50; 100; 150; 200; 250];
    xi_tab = [0.15; 0.2; 0.25; 0.3; 0.35; 0.4];
    int_corrFactor = interpolate((E_tab,xi_tab),correction_factors,Gridded(Linear()))
    p = int_corrFactor((R1/alpha)^(1/p0),xi)
    r_k = zeros(n) #ranges
    e_k = zeros(n) #energies
    w_k = zeros(n) #weights
    R0 = (1-xi)*R1

    r_k[1] = (1-(1-1/n)*xi)*R1
    w_k[1] = 1-(1-1/(2*n))^(1-1/p)
    for k=2:n-1
        r_k[k] = (1-(1-k/n)*xi)*R1
        w_k[k] = (1-(1/n)*(k-1/2))^(1-1/p) - (1-(1/n)*(k+1/2))^(1-1/p)
    end
    r_k[n] = R1
    w_k[n] = (1/(2*n))^(1-1/p)
    e_k = (r_k./alpha).^(1/p0)
    return r_k, e_k, w_k
end

function dqage(
    f::Ptr{Cvoid}, a::Float64, b::Float64,
    epsabs::Float64, epsrel::Float64,
    key::Int32, limit::Int32
)
    # Preallocate arrays and outputs
    result = Ref{Float64}(0.0)
    abserr = Ref{Float64}(0.0)
    neval = Ref{Int32}(0)
    ier = Ref{Int32}(0)
    alist = Vector{Float64}(undef, limit)
    blist = Vector{Float64}(undef, limit)
    rlist = Vector{Float64}(undef, limit)
    elist = Vector{Float64}(undef, limit)
    iord = Vector{Int32}(undef, limit)
    last = Ref{Int32}(0)

    # Call the Fortran function using ccall
    ccall(
        (:dqage_, "tracer/libquadpack"), Cvoid,
        (Ptr{Cvoid}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64},
         Ref{Int32}, Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Int32},
         Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Int32}, Ref{Int32}),
        f, a, b, epsabs, epsrel,
        key, limit, result, abserr, neval, ier,
        alist, blist, rlist, elist, iord, last
    )

    # Return results
    return (
        result[], abserr[], neval[], ier[],
        alist, blist, rlist, elist, iord, last[]
    )
end

function dqags(f::Ptr{Cvoid}, a::Float64, b::Float64, epsabs::Float64=1e-9, epsrel::Float64=1e-9)
    # Define limit (maximum number of subintervals for adaptive refinement)
    limit = 500

    # Allocate memory for outputs
    result = Ref{Float64}(0.0)  # Integral result
    abserr = Ref{Float64}(0.0)  # Estimate of absolute error
    neval = Ref{Int32}(0)       # Number of function evaluations
    ier = Ref{Int32}(0)         # Error flag
    last = Ref{Int32}(0)        # Number of subintervals used

    # Workspace arrays
    lenw = 4 * limit
    iwork = zeros(Int32, limit)  # Integer workspace
    work = zeros(Float64, lenw)  # Floating-point workspace

    # Call dqags via ccall
    ccall(
        (:dqags_, "tracer/libquadpack"), # Ensure libquadpack is linked correctly
        Cvoid,
        (Ptr{Cvoid}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64},
         Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int32},
         Ref{Int32}, Ref{Int32}, Ptr{Int32}, Ptr{Float64}),
        f, # Function pointer
        a, b, epsabs, epsrel,
        result, abserr, neval, ier,
        limit, lenw, last, iwork, work
    )

    return (result[], abserr[], neval[], ier[])
end

function matComp(HU::Array{T,1},entryType::String="HU") where {T<:AbstractFloat}
    #this is equivalent to the mat_comp.f90 class in tracer (adapted to requirements of DLRA code)
    #computes density and material composition for given HU CT values
    HU_min = -1000
    HU_max = +1600

    #material info
    no_comp_nuclides = 12
    H_mat  =  1
    C_mat  =  2
    N_mat  =  3
    O_mat  =  4
    Na_mat =  5
    Mg_mat =  6
    P_mat  =  7
    S_mat  =  8
    Cl_mat =  9
    Ar_mat = 10
    K_mat  = 11
    Ca_mat = 12

    rho_H  = 8.3748E-05
    rho_C  = 2.0
    rho_N  = 0.0011652
    rho_O  = 0.00133151
    ho_Na = 0.971
    rho_Mg = 1.74
    rho_P  = 2.2
    rho_S  = 2.0
    rho_Cl = 0.00299473
    rho_Ar = 0.00166201
    rho_K  = 0.862
    rho_Ca = 1.55

    #Atomic numbers
    Z_array = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
    #Atomic weights
    Mol_weights = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078]
    #Ionization energies
    I_eV  = vcat(19.0, (11.2 .+ 11.7.*Z_array[2:6]), (52.8 .+ 8.71.*Z_array[7:12]))
    I_MeV = 1.0E-6 .* I_eV
    #Ln of ionization energies
    ln_I_MeV = log.(I_MeV)

    rho_air = 1.21E-3
    rho_adipose = 0.93

    #compute comp. vector
    comp_vector = zeros(no_comp_nuclides,length(HU))

    for  i=1:length(HU)
        if     ( (HU[i] >= -1000)&&(HU[i] <=  -950) ) 
            comp_vector[H_mat ,i] =  0.0
            comp_vector[C_mat ,i] =  0.0
            comp_vector[N_mat ,i] = 75.5
            comp_vector[O_mat ,i] = 23.3
            comp_vector[Na_mat ,i] =  0.0
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.0
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  1.3
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >   -950)&&(HU[i] <=  -120) ) 
            comp_vector[H_mat ,i] = 10.3
            comp_vector[C_mat ,i] = 10.5
            comp_vector[N_mat ,i] =  3.1
            comp_vector[O_mat ,i] = 74.9
            comp_vector[Na_mat ,i] =  0.2
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.2
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.3
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.2
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >   -120)&&(HU[i] <=   -83) ) 
            comp_vector[H_mat ,i] = 11.6
            comp_vector[C_mat ,i] = 68.1
            comp_vector[N_mat ,i] =  0.2
            comp_vector[O_mat ,i] = 19.8
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    -83)&&(HU[i] <=   -53) ) 
            comp_vector[H_mat ,i] = 11.3
            comp_vector[C_mat ,i] = 56.7
            comp_vector[N_mat ,i] =  0.9
            comp_vector[O_mat ,i] = 30.8
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    -53)&&(HU[i] <=   -23) ) 
            comp_vector[H_mat ,i] = 11.0
            comp_vector[C_mat ,i] = 45.8
            comp_vector[N_mat ,i] =  1.5
            comp_vector[O_mat ,i] = 41.1
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    -23)&&(HU[i] <=    +7) ) 
            comp_vector[H_mat ,i] = 10.8
            comp_vector[C_mat ,i] = 35.6
            comp_vector[N_mat ,i] =  2.2
            comp_vector[O_mat ,i] = 50.9
            comp_vector[Na_mat ,i] =  0.0
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >     +7)&&(HU[i] <=   +18) ) 
            comp_vector[H_mat ,i] = 10.6
            comp_vector[C_mat ,i] = 28.4
            comp_vector[N_mat ,i] =  2.6
            comp_vector[O_mat ,i] = 57.8
            comp_vector[Na_mat ,i] =  0.0
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    +18)&&(HU[i] <=   +80) ) 
            comp_vector[H_mat ,i] = 10.3
            comp_vector[C_mat ,i] = 13.4
            comp_vector[N_mat ,i] =  3.0
            comp_vector[O_mat ,i] = 72.3
            comp_vector[Na_mat ,i] =  0.2
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.2
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.2
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.2
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >    +80)&&(HU[i]<=  +120) ) 
            comp_vector[H_mat ,i] =  9.4
            comp_vector[C_mat ,i] = 20.7
            comp_vector[N_mat ,i] =  6.2
            comp_vector[O_mat ,i] = 62.2
            comp_vector[Na_mat ,i] =  0.6
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  0.0
            comp_vector[S_mat ,i] =  0.6
            comp_vector[Cl_mat ,i] =  0.3
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] =  0.0
        elseif ( (HU[i] >   +120)&&(HU[i] <=  +200) ) 
            comp_vector[H_mat ,i] =  9.5
            comp_vector[C_mat ,i] = 45.5
            comp_vector[N_mat ,i] =  2.5
            comp_vector[O_mat ,i] = 35.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  2.1
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  4.5
        elseif ( (HU[i]>   +200)&&(HU[i] <=  +300) ) 
            comp_vector[H_mat ,i] =  8.9
            comp_vector[C_mat ,i] = 42.3
            comp_vector[N_mat ,i] =  2.7
            comp_vector[O_mat ,i] = 36.3
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  3.0
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  6.4
        elseif ( (HU[i] >   +300)&&(HU[i] <=  +400) ) 
            comp_vector[H_mat ,i] =  8.2
            comp_vector[C_mat ,i] = 39.1
            comp_vector[N_mat ,i] =  2.9
            comp_vector[O_mat ,i] = 37.2
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.0
            comp_vector[P_mat ,i] =  3.9
            comp_vector[S_mat ,i] =  0.1
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.1
            comp_vector[Ca_mat ,i] =  8.3
        elseif ( (HU[i] >   +400)&&(HU[i] <=  +500) ) 
            comp_vector[H_mat ,i] =  7.6
            comp_vector[C_mat ,i] = 36.1
            comp_vector[N_mat ,i] =  3.0
            comp_vector[O_mat ,i] = 38.0
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  4.7
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.1
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 10.1
        elseif ( (HU[i] >   +500)&&(HU[i] <=  +600) ) 
            comp_vector[H_mat ,i] =  7.1
            comp_vector[C_mat ,i] = 33.5
            comp_vector[N_mat ,i] =  3.2
            comp_vector[O_mat ,i] = 38.7
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  5.4
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 11.7
        elseif ( (HU[i]>   +600)&&(HU[i] <=  +700) ) 
            comp_vector[H_mat ,i] =  6.6
            comp_vector[C_mat ,i] = 31.0
            comp_vector[N_mat ,i] =  3.3
            comp_vector[O_mat ,i] = 39.4
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  6.1
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 13.2
        elseif ( (HU[i] >   +700)&&(HU[i] <=  +800) ) 
            comp_vector[H_mat ,i] =  6.1
            comp_vector[C_mat ,i] = 28.7
            comp_vector[N_mat ,i] =  3.5
            comp_vector[O_mat ,i] = 40.0
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.1
            comp_vector[P_mat ,i] =  6.7
            comp_vector[S_mat ,i] =  0.2
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 14.6
        elseif ( (HU[i]>   +800)&&(HU[i]<=  +900) ) 
            comp_vector[H_mat ,i] =  5.6
            comp_vector[C_mat ,i] = 26.5
            comp_vector[N_mat ,i] =  3.6
            comp_vector[O_mat ,i] = 40.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  7.3
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 15.9
        elseif ( (HU[i] >   +900)&&(HU[i] <= +1000) ) 
            comp_vector[H_mat ,i] =  5.2
            comp_vector[C_mat ,i] = 24.6
            comp_vector[N_mat ,i] =  3.7
            comp_vector[O_mat ,i] = 41.1
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  7.8
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 17.0
        elseif ( (HU[i]>  +1000)&&(HU[i]<= +1100) ) 
            comp_vector[H_mat ,i] =  4.9
            comp_vector[C_mat ,i] = 22.7
            comp_vector[N_mat ,i] =  3.8
            comp_vector[O_mat ,i] = 41.6
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  8.3
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 18.1
        elseif ( (HU[i]>  +1100)&&(HU[i]<= +1200) ) 
            comp_vector[H_mat ,i] =  4.5
            comp_vector[C_mat ,i] = 21.0
            comp_vector[N_mat ,i] =  3.9
            comp_vector[O_mat ,i] = 42.0
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  8.8
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 19.2
        elseif ( (HU[i]>  +1200)&&(HU[i]<= +1300) ) 
            comp_vector[H_mat ,i] =  4.2
            comp_vector[C_mat ,i] = 19.4
            comp_vector[N_mat ,i] =  4.0
            comp_vector[O_mat ,i] = 42.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  9.2
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 20.1
        elseif ( (HU[i] >  +1300)&&(HU[i] <= +1400) ) 
            comp_vector[H_mat ,i] =  3.9
            comp_vector[C_mat ,i] = 17.9
            comp_vector[N_mat ,i] =  4.1
            comp_vector[O_mat ,i] = 42.9
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] =  9.6
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 21.0
        elseif ( (HU[i]>  +1400)&&(HU[i] <= +1500) ) 
            comp_vector[H_mat ,i] =  3.6
            comp_vector[C_mat ,i] = 16.5
            comp_vector[N_mat ,i] =  4.2
            comp_vector[O_mat ,i] = 43.2
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] = 10.0
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 21.9
        elseif ( (HU[i] >  +1500)&&(HU[i] <= +1600) ) 
            comp_vector[H_mat ,i] =  3.4
            comp_vector[C_mat ,i] = 15.5
            comp_vector[N_mat ,i] =  4.2
            comp_vector[O_mat ,i] = 43.5
            comp_vector[Na_mat ,i] =  0.1
            comp_vector[Mg_mat ,i] =  0.2
            comp_vector[P_mat ,i] = 10.3
            comp_vector[S_mat ,i] =  0.3
            comp_vector[Cl_mat ,i] =  0.0
            comp_vector[Ar_mat ,i] =  0.0
            comp_vector[K_mat ,i] =  0.0
            comp_vector[Ca_mat ,i] = 22.5
        else
            println("CT_value out of range")
        end
    end
    return comp_vector
end

function safe_svd(S)
    U, D, V = nothing, nothing, nothing  # Initialize variables
    try
        U, D, V = svd(Matrix(S))  # Try standard SVD first
    catch e
        @warn "SVD failed, falling back to LAPACK gesvd!" exception=(e, catch_backtrace())
        U, D_vec, Vt = LinearAlgebra.LAPACK.gesvd!('A', 'A', copy(S))  # Full SVD
        D = Diagonal(D_vec)  # Convert singular values to a diagonal matrix
        V = Vt'  # LAPACK returns Vt, so transpose it to get V
    end
    return U, D, V
end

# using MatrixEquations

# function implicit_euler_sylvester(L, X, Xp, Sigma_t, G, dt)
#     A = I - dt * (Xp * Sigma_t * X)  # r x r
#     B = dt * G  # m x m
#     C = L  # r x m

#     return sylvester(A, B, C)  # Solve A*L + L*B = C
# end

function get_beamAtEntry(box_min, box_max, origin, direction, sigma)
    #determine plane through which beam enters domain and parameters of Gaussian beam dist in this plane
    dir_x, dir_y, dir_z = direction
    x_min, y_min, z_min = box_min
    x_max, y_max, z_max = box_max
    x0, y0, z0 = origin

    # Dictionary to store intersection times for each face
    faces = Dict(
        "Front"  => Inf, "Back"  => Inf,
        "Left"   => Inf, "Right" => Inf,
        "Bottom" => Inf, "Top"   => Inf
    )

    # Compute t-values for each face and check if intersection is valid
    if dir_z != 0
        t_f = (z_min - z0) / dir_z  # Front face
        t_b = (z_max - z0) / dir_z  # Back face
        if x_min <= x0 + t_f * dir_x <= x_max && y_min <= y0 + t_f * dir_y <= y_max && t_f > 0
            faces["Front"] = t_f
        end
        if x_min <= x0 + t_b * dir_x <= x_max && y_min <= y0 + t_b * dir_y <= y_max && t_b > 0
            faces["Back"] = t_b
        end
    end

    if dir_x != 0
        t_l = (x_min - x0) / dir_x  # Left face
        t_r = (x_max - x0) / dir_x  # Right face
        if y_min <= y0 + t_l * dir_y <= y_max && z_min <= z0 + t_l * dir_z <= z_max && t_l > 0
            faces["Left"] = t_l
        end
        if y_min <= y0 + t_r * dir_y <= y_max && z_min <= z0 + t_r * dir_z <= z_max && t_r > 0
            faces["Right"] = t_r
        end
    end

    if dir_y != 0
        t_bo = (y_min - y0) / dir_y  # Bottom face
        t_t = (y_max - y0) / dir_y  # Top face
        if x_min <= x0 + t_bo * dir_x <= x_max && z_min <= z0 + t_bo * dir_z <= z_max && t_bo > 0
            faces["Bottom"] = t_bo
        end
        if x_min <= x0 + t_t * dir_x <= x_max && z_min <= z0 + t_t * dir_z <= z_max && t_t > 0
            faces["Top"] = t_t
        end
    end

    # Find the closest valid intersection
    entry_plane = argmin(faces)
    t_entry = faces[entry_plane]

    # Compute mean intersection point
    mean_x = x0 + t_entry * dir_x
    mean_y = y0 + t_entry * dir_y
    mean_z = z0 + t_entry * dir_z

    # Determine projection and standard deviation stretching
    plane_normals = Dict(
        "Front"  => (0.0, 0.0, -1.0),
        "Back"   => (0.0, 0.0,  1.0),
        "Left"   => (-1.0, 0.0, 0.0),
        "Right"  => (1.0, 0.0, 0.0),
        "Bottom" => (0.0, -1.0, 0.0),
        "Top"    => (0.0,  1.0, 0.0)
    )
    normal = plane_normals[entry_plane]
    normal = collect(normal)

    # Rotate Gaussian dist.
    rotM = rotate_bev_to_xyz(direction) #get_rotMatrix([0,0,1],direction) #variances are defined in beams eye view (z-axis is beam dir)
    rotated_Σ = abs.(rotM*sigma)

    # Find the two principal axes in the entry plane
    if normal ≈ [0, 0, 1] || normal ≈ [0, 0, -1]  # Front/Back (XY Plane)
        mean_entry = [mean_x, mean_y] #mean_entry = rotated_mean[1:2]  # Take (x, y)
        sigma_entry = [rotated_Σ[1], rotated_Σ[2]] # Extract variances
    elseif normal ≈ [1, 0, 0] || normal ≈ [-1, 0, 0]  # Left/Right (YZ Plane)
        mean_entry = [mean_y, mean_z]#mean_entry = rotated_mean[2:3]  # Take (y, z)
        sigma_entry = [rotated_Σ[2], rotated_Σ[3]]
    elseif normal ≈ [0, 1, 0] || normal ≈ [0, -1, 0]  # Top/Bottom (XZ Plane)
        mean_entry = [mean_x, mean_z]#mean_entry = rotated_mean[[1,3]]  # Take (x, z)
        sigma_entry = [rotated_Σ[1], rotated_Σ[3]]
    else
        error("Invalid normal vector")
    end

    #mean_entry = [mean_x, mean_y, mean_z]
    #sigma_entry = rotated_Σ 

    return entry_plane, mean_entry, sigma_entry
end

function rotate_bev_to_xyz(beam_direction)
    beam_direction = normalize(beam_direction)  # Ensure it's a unit vector
    k = cross([0, 0, 1], beam_direction)  # Rotation axis
    if norm(k) ≈ 0  # If beam direction is already aligned, return identity
        return I(3)
    end
    
    k = normalize(k)  # Normalize rotation axis
    θ = acos(clamp(dot([0, 0, 1], beam_direction), -1.0, 1.0))  # Rotation angle

    K = [  0     -k[3]   k[2];
           k[3]   0     -k[1];
          -k[2]   k[1]   0   ]  # Skew-symmetric cross-product matrix

    R = I(3) + sin(θ) * K + (1 - cos(θ)) * (K * K)  # Rodrigues' rotation formula
    return R
end

function rotate_xyz_to_bev(beam_direction)
    beam_direction = normalize(beam_direction)  # Ensure it's a unit vector
    k = cross(beam_direction,[0, 0, 1])  # Rotation axis
    if norm(k) ≈ 0  # If beam direction is already aligned, return identity
        return I(3)
    end
    
    k = normalize(k)  # Normalize rotation axis
    θ = acos(clamp(dot(beam_direction,[0, 0, 1]), -1.0, 1.0))  # Rotation angle

    K = [  0     -k[3]   k[2];
           k[3]   0     -k[1];
          -k[2]   k[1]   0   ]  # Skew-symmetric cross-product matrix

    R = I(3) + sin(θ) * K + (1 - cos(θ)) * (K * K)  # Rodrigues' rotation formula
    return R
end


function rotateAxis(Theta,axis)
    #Rotation matrices
    Theta=Theta*pi/180;#angle in rad
    if axis == "x"
        rotTheta = [1 0 0; 0 cos(Theta) -sin(Theta); 0 sin(Theta) cos(Theta)]
    elseif axis == "y"
        rotTheta=[cos(Theta) 0 sin(Theta); 0 1 0; -sin(Theta) 0 cos(Theta)]
    else
        rotTheta=[cos(Theta) -sin(Theta) 0; sin(Theta) cos(Theta) 0; 0 0 1];
    end
    return rotTheta
end

function quad_generalGauss2D(mu::Vector, Sigma::Matrix, num_points::Int)
    dim = length(mu)  # Dimension of space

    # Compute Cholesky decomposition of Sigma (for transformation)
    L = cholesky(Sigma).L

    # Generate Gauss-Hermite quadrature points and weights for each dimension
    points_1D, weights_1D = gausshermite(num_points)  # Standard normal N(0,1)

    # Create grid of quadrature points
    num_total = num_points^dim  # Total number of points
    quad_points = zeros(dim, num_total)  # Store as a matrix (each column = 1 point)
    quad_weights = zeros(num_total)  # Store weights as a vector

    # Fill matrix with transformed quadrature points and compute weights
    index = 1
    for i in 1:num_points, j in 1:num_points
        # Quadrature point in standard normal space
        p_std = [points_1D[i], points_1D[j]]  # Only works for 2D

        # Transform to general Gaussian N(mu, Sigma)
        quad_points[:, index] = mu + L * p_std

        # Compute transformed weight
        quad_weights[index] = weights_1D[i] * weights_1D[j] * sqrt(det(2π * Sigma))

        index += 1
    end

    return quad_points, quad_weights
end

function generate_3D_grid(box_min, box_max, grid_size)
    x = range(box_min[1], box_max[1], length=grid_size[1]+2)[2:end-1]
    y = range(box_min[2], box_max[2], length=grid_size[2]+2)[2:end-1]
    z = range(box_min[3], box_max[3], length=grid_size[3]+2)[2:end-1]

    grid_points = [ [xi, yi, zi] for xi in x for yi in y for zi in z ]
    return hcat(grid_points...)  # Convert to 3×N matrix
end
