# rectilinear_csda.jl
# 3D rectilinear ray tracer + solver for:
#   ∂φ/∂z + Σ_a φ = ∂/∂E ( S φ ) + 1/2 ∂/∂E ( T ∂φ/∂E )
#
# - Gaussian initial energy distribution (mean, sigma, total_intensity)
# - Uniform energy grid
# - Implicit Euler in z, central differences in energy (tridiagonal solve)
# - Dirichlet φ(Emin)=0 (absorbing low-energy boundary), Neumann-like/backward at Emax
# - Returns φ(E) sampled along ray (positions and energies)

module RaytracerCSD

export RectMesh, Material, trace_ray_csda

using LinearAlgebra

# -------------------------
# Types
# -------------------------
struct RectMesh
    xedges::Vector{Float64}  # length Nx+1
    yedges::Vector{Float64}  # length Ny+1
    zedges::Vector{Float64}  # length Nz+1
    matids::Array{Int,3}     # size (Nx,Ny,Nz)
end

struct Material
    id::Int
    sigma_t::Function   # Σ_t(E) total scattering cross section (outscattering + absorption)
    S::Function         # stopping power S(E) (sign per your convention)
    T::Function         # straggling coefficient T(E) (>=0)
end

const EPS = 1e-14

function find_cell(edges::Vector{Float64}, coord::Float64)
    i = searchsortedlast(edges, coord)
    if i == 0 || i == length(edges)
        return 0
    else
        return i
    end
end

# Ray-box entry t calculation
function ray_box_entry(origin::NTuple{3,Float64}, dir::NTuple{3,Float64},
                       xmin,xmax,ymin,ymax,zmin,zmax)
    ox,oy,oz = origin
    dx,dy,dz = dir
    tmin = -Inf; tmax = Inf
    for (o,d,emin,emax) in ((ox,dx,xmin,xmax),(oy,dy,ymin,ymax),(oz,dz,zmin,zmax))
        if abs(d) < EPS
            if !(o >= emin - EPS && o <= emax + EPS)
                return nothing  # ray parallel and outside slab
            end
        else
            t1 = (emin - o)/d
            t2 = (emax - o)/d
            ta = min(t1,t2); tb = max(t1,t2)
            tmin = max(tmin, ta)
            tmax = min(tmax, tb)
            if tmin > tmax
                return nothing
            end
        end
    end
    if tmax < 0
        return nothing
    end
    return max(tmin, 0.0), tmax
end

# Marching through mesh -> returns segments 
function march_ray(mesh::RectMesh, origin::NTuple{3,Float64}, direction::NTuple{3,Float64};
                   max_t::Float64 = 1e9)

    # normalize direction (unit vector)
    dvec = Float64.(direction)
    dn = norm(dvec)
    @assert dn > 0 "direction must be non-zero"
    dir_unit = dvec ./ dn
    dx,dy,dz = dir_unit
    ox,oy,oz = Float64.(origin)

    xedges,yedges,zedges = mesh.xedges, mesh.yedges, mesh.zedges
    Nx = length(xedges)-1; Ny = length(yedges)-1; Nz = length(zedges)-1

    # If starting outside, compute entry t
    bb_entry = ray_box_entry((ox,oy,oz), (dx,dy,dz),
                             xedges[1], xedges[end], yedges[1], yedges[end], zedges[1], zedges[end])
    if bb_entry === nothing
        return []  # ray misses mesh
    end
    t_entry, t_exit = bb_entry
    # move origin to just inside entry
    ox += dx * (t_entry + 1e-12)
    oy += dy * (t_entry + 1e-12)
    oz += dz * (t_entry + 1e-12)

    ix = find_cell(xedges, ox)
    iy = find_cell(yedges, oy)
    iz = find_cell(zedges, oz)
    if ix == 0 || iy == 0 || iz == 0
        return []
    end

    segments = Vector{Dict{Symbol,Any}}()
    # marching loop: compute t to next plane for current cell directly
    current_pos = (ox,oy,oz)
    total_t = 0.0
    while ix >= 1 && ix <= Nx && iy >= 1 && iy <= Ny && iz >= 1 && iz <= Nz && total_t < max_t
        x0 = xedges[ix]; x1 = xedges[ix+1]
        y0 = yedges[iy]; y1 = yedges[iy+1]
        z0 = zedges[iz]; z1 = zedges[iz+1]

        tx = abs(dx) < EPS ? Inf : minimum(( (x0 - current_pos[1]) / dx, (x1 - current_pos[1]) / dx ))
        tx2 = abs(dx) < EPS ? Inf : maximum(( (x0 - current_pos[1]) / dx, (x1 - current_pos[1]) / dx ))
        # we want positive exit distance along dir_unit; compute t to the plane in forward direction:
        tx_fwd = abs(dx) < EPS ? Inf : ((dx > 0) ? (x1 - current_pos[1]) / dx : (x0 - current_pos[1]) / dx)
        ty_fwd = abs(dy) < EPS ? Inf : ((dy > 0) ? (y1 - current_pos[2]) / dy : (y0 - current_pos[2]) / dy)
        tz_fwd = abs(dz) < EPS ? Inf : ((dz > 0) ? (z1 - current_pos[3]) / dz : (z0 - current_pos[3]) / dz)

        # choose nearest positive t (with small tolerance)
        t_to_exit = minimum((tx_fwd, ty_fwd, tz_fwd))
        if !isfinite(t_to_exit) || t_to_exit < 0
            break
        end
        # cap by remaining max_t
        t_seg = min(t_to_exit, max_t - total_t)
        seg_entry = current_pos
        seg_exit = (current_pos[1] + dx * t_seg,
                    current_pos[2] + dy * t_seg,
                    current_pos[3] + dz * t_seg)
        seg_len = t_seg  # dir_unit is normalized -> param equals Euclidean length

        push!(segments, Dict(:entry => seg_entry,
                             :exit  => seg_exit,
                             :length => seg_len,
                             :i => ix, :j => iy, :k => iz,
                             :matid => mesh.matids[ix,iy,iz]))

        # advance
        total_t += t_seg
        # move current_pos slightly beyond exit to avoid numerical re-hit
        current_pos = (seg_exit[1] + dx * 1e-12,
                       seg_exit[2] + dy * 1e-12,
                       seg_exit[3] + dz * 1e-12)
        ix = find_cell(xedges, current_pos[1])
        iy = find_cell(yedges, current_pos[2])
        iz = find_cell(zedges, current_pos[3])
    end

    return segments
end

# -------------------------
# Tridiagonal solver (Thomas)
# a: lower diag (a[1]=0), b: main diag, c: upper (c[n]=0), rhs d
function tdma!(a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64}, d::Vector{Float64})
    n = length(b)
    # copy to avoid destroying inputs (we operate in place on d)
    cp = similar(c)
    bp = similar(b)
    bp[1] = b[1]
    cp[1] = (n>1) ? c[1]/bp[1] : 0.0
    d[1] /= bp[1]
    for i in 2:n-1
        bp[i] = b[i] - a[i]*cp[i-1]
        cp[i] = c[i] / bp[i]
        d[i] = (d[i] - a[i]*d[i-1]) / bp[i]
    end
    if n > 1
        bp[n] = b[n] - a[n]*cp[n-1]
        d[n] = (d[n] - a[n]*d[n-1]) / bp[n]
    end
    # back substitution
    x = similar(d)
    x[n] = d[n]
    for i in n-1:-1:1
        x[i] = d[i] - cp[i]*x[i+1]
    end
    return x
end

function advance_phi_upwind(phi_in::Vector{Float64},
                            Δz::Float64,
                            E::Vector{Float64},
                            Sfun::Function,
                            Tfun::Function,
                            Sigma_fun::Function)

    N = length(E)
    dE = E[2] - E[1]

    Svec = [Sfun(E[i]) for i in 1:N]
    Tvec = [Tfun(E[i]) for i in 1:N]
    Sigvec = [Sigma_fun(E[i]) for i in 1:N]
    T_half = [0.5*(Tvec[i]+Tvec[i+1]) for i in 1:N-1]

    a = zeros(N); b = zeros(N); c = zeros(N)

    # Dirichlet at Emin
    b[1] = 0.0

    for i in 2:N-1

        if Svec[i] >= 0
            # backward diff
            drift =  Svec[i] / dE
            c[i]  = -drift
            b[i] +=  drift
        else
            # forward diff
            drift = -Svec[i] / dE
            a[i]  = -drift
            b[i] +=  drift
        end
        # diffusion
        a[i] += 0.5 * T_half[i] / dE^2
        c[i] += 0.5 * T_half[i-1] / dE^2
        b[i] += -Sigvec[i] - 0.5*(T_half[i]+T_half[i-1])/dE^2
    end

    i = N
    if Svec[i] >= 0
        c[i] = -Svec[i]/dE
        b[i] +=  Svec[i]/dE
    else
        b[i] += -Svec[i]/dE
    end
    b[i] += -Sigvec[i] - 0.5*T_half[N-1]/dE^2
    c[i] +=  0.5*T_half[N-1]/dE^2

    al = [-Δz*c[i] for i in 1:N]
    cu = [-Δz*a[i] for i in 1:N]
    bv = [1.0 - Δz*b[i] for i in 1:N]

    # Dirichlet at Emin
    bv[1] = 1.0; al[1]=0.0; cu[1]=0.0

    rhs = copy(phi_in)
    rhs[1] = 0.0

    return tdma!(al, bv, cu, rhs)
end

#main tracing fct.
function trace_ray_csda(mesh::RectMesh, materials::Dict{Int,Material},
                        origin::NTuple{3,Real}, direction::NTuple{3,Real};
                        E_range::Tuple{Float64,Float64,Int} = (0.0, 5.0, 201),
                        gaussian::Tuple{Float64,Float64,Float64} = (2.5, 0.1, 1.0),
                        samples_per_segment::Int = 1,
                        max_substep_dz::Float64 = 0.5)

    Emin,Emax,N = E_range
    E = range(Emin, stop=Emax, length=N) |> collect
    dE = E[2]-E[1]

    μ0,σ0,I0 = gaussian
    raw = [exp(-(E[i]-μ0)^2/(2σ0^2)) for i in 1:N]
    norm = sum(raw)*dE
    phi0 = [I0*raw[i]/norm for i in 1:N]

    segs = march_ray(mesh, Float64.(origin), Float64.(direction))
    if isempty(segs)
        return Dict(:energies=>E,:positions=>[],:distances=>Float64[],
                    :phi=>Array{Float64,2}(undef,0,N),
                    :segments=>segs,
                    :deposition=>zeros((size(mesh.matids)...,N)))
    end

    positions = [segs[1][:entry]]
    distances = [0.0]
    phi_list = [copy(phi0)]

    # --- full 4D deposition: (i,j,k,e) ---
    deposition = zeros(Float64, (size(mesh.matids)..., N))

    cumulative_z = 0.0
    phi = copy(phi0)

    for seg in segs
        seg_len = seg[:length]
        matid = seg[:matid]
        mat = materials[matid]
        i,j,k = seg[:i], seg[:j], seg[:k]

        m = max(samples_per_segment, Int(ceil(seg_len/max_substep_dz)))
        dz = seg_len/m

        for s in 1:m
            phi_before = copy(phi)
            phi = advance_phi_upwind(phi, dz, E, mat.S, mat.T, mat.sigma_t)
            cumulative_z += dz

            # particle density loss per energy bin
            dphi = phi_before .- phi
            deposition[i,j,k,:] .+= dphi

            pos = (segs[1][:entry][1] + direction[1]*cumulative_z,
                   segs[1][:entry][2] + direction[2]*cumulative_z,
                   segs[1][:entry][3] + direction[3]*cumulative_z)
            push!(positions,pos)
            push!(distances,cumulative_z)
            push!(phi_list,copy(phi))
        end
    end

    npos = length(phi_list)
    PHI = zeros(npos,N)
    for i in 1:npos
        PHI[i,:] = phi_list[i]
    end

    return Dict(:energies=>E,
                :positions=>positions,
                :distances=>distances,
                :phi=>PHI,
                :segments=>segs,
                :deposition=>deposition)
end

end # module
