# Set-up and solvers for proton pencil beams with uncertain stopping-power ratio (CT calibration) and lateral setup error
#
# A case is defined by four structs
#   ProtonBeam         initial energy, energy spread, lateral position and width (beam enters at x = a along +x)
#   Phantom            geometry: list of ProtonMaterials, material index and nominal density per cell
#   ProtonUncertainty  ξ: SPR (density) error, ρ(x;ξ) = ρ₀(x)(1 + σ_m(x) ξ), correlated over all cells with a
#                      material-specific σ_m; η: lateral setup error, y_b = y0 + σSetup η
#   ProtonNumerics     energy cut-off, FP correction order, tolerances, CFL number

using LinearAlgebra
using FastGaussQuadrature
using SparseArrays

# case definition

Base.@kwdef struct ProtonBeam
    E0::Float64 = 90.0          # nominal initial kinetic energy [MeV]
    σE_rel::Float64 = 0.01      # relative energy spread (Gaussian)
    y0::Float64 = 4.0           # nominal lateral position [cm]
    w::Float64 = 0.5            # lateral width (std. dev.) [cm]
end

struct Phantom
    materials::Vector{ProtonMaterial}
    matIndex::Vector{Int}       # material of each cell (index into materials, vectorIndex ordering)
    ρ0::Vector{Float64}         # nominal density of each cell [g/cm³]
end

# phantom from per-cell arrays (nx × ny, first index x), e.g. obtained from a CT slice
function Phantom(s::Settings, materials::Vector{ProtonMaterial}, matIndex::AbstractMatrix{<:Integer}, ρ0::AbstractMatrix{<:Real})
    nx = s.NCells; ny = nx
    @assert size(matIndex) == (nx, ny) && size(ρ0) == (nx, ny)
    @assert all(1 .<= matIndex .<= length(materials)) && all(ρ0 .> 0)
    mi = zeros(Int, nx * ny); ρ = zeros(nx * ny)
    for i = 1:nx, j = 1:ny
        mi[vectorIndex(ny, i, j)] = matIndex[i, j]; ρ[vectorIndex(ny, i, j)] = ρ0[i, j]
    end
    return Phantom(materials, mi, ρ)
end

# phantom from functions of the cell centre: material(x, y) -> index, density(x, y) -> ρ₀ [g/cm³]
Phantom(s::Settings, materials::Vector{ProtonMaterial}, material::Function, density::Function) =
    Phantom(s, materials, [material(x, y) for x in s.xMid, y in s.xMid], [density(x, y) for x in s.xMid, y in s.xMid])

# example: water phantom with a rectangular insert (default: bone, HU ≈ 950 in TITUS HUtoDensity/matComp)
function waterPhantomWithInsert(s::Settings; insX=[1.5, 3.0], insY=[4.0, 6.5], ρInsert=1.579,
                                insert=ProtonMaterial("bone", COMP_BONE), water=ProtonMaterial("water", COMP_WATER))
    inside(x, y) = insX[1] <= x <= insX[2] && insY[1] <= y <= insY[2]
    return Phantom(s, [water, insert], (x, y) -> inside(x, y) ? 2 : 1, (x, y) -> inside(x, y) ? ρInsert : 1.0)
end

Base.@kwdef struct ProtonUncertainty
    σSPR::Dict{String,Float64} = Dict("water" => 0.016, "bone" => 0.024) # 1σ relative SPR error per material name
    σSetup::Float64 = 0.3       # 1σ lateral setup error [cm]
    distξ::Symbol = :gaussian   # distribution of ξ and η (see parameterQuadrature)
    distη::Symbol = :gaussian
end

Base.@kwdef mutable struct ProtonNumerics
    Ecut::Float64 = 1.0         # cut-off energy [MeV]
    Nc::Int = 0                 # FP correction order; ≤ 0 means Nc = P_N order 
    nSamples::Int = 601         # initial-energy samples for the uncollided characteristics
    tolSource::Float64 = 1e-6   # relative tolerance for compressing the source per energy step
    cflE::Float64 = 0.5         # CFL number for the energy step, dE = cflE Δx min S
end

struct ProtonProblem
    beam::ProtonBeam
    phantom::Phantom
    unc::ProtonUncertainty
    num::ProtonNumerics
    σm::Vector{Float64}            # SPR uncertainty of each material
    ξ::Vector{Float64}; pξ::Vector{Float64}   # quadrature nodes and probability weights
    η::Vector{Float64}; pη::Vector{Float64}
    Egrid::Vector{Float64}         # decreasing energy grid
    dE::Float64
    aΩ::Vector{Float64}            # angular moments of the first-collision source (per unit ψ̃_u)
    terms::Vector{Any}             # (spatial, angle, ξ, η, coefficient E -> c(E))
    profiles::Array{Float64,4}     # ψ̃_u along x: nx × nE × row class × nξ
    rowClass::Vector{Int}          # row class of each y-row (0: not reached by the beam)
    lateral::Matrix{Float64}       # lateral beam profile g(y_j; η_l), ny × nη
end

# angular quantities

# real spherical harmonics Y_ℓm(Ω) in the moment ordering of PNSystem, obtained from the recursion
# Ωₓ Y = Aₓ Y (rows ℓ < N) etc., which the P_N matrices satisfy exactly
function sphericalHarmonicsAt(pn::PNSystem, Ω::Vector{Float64})
    N = pn.N; nΩ = pn.nTotalEntries
    rows = [i for i in 0:nΩ-1 if floor(Int, sqrt(i)) < N] .+ 1
    M = vcat(pn.Ax[rows, :] - Ω[1] * I(nΩ)[rows, :], pn.Ay[rows, :] - Ω[2] * I(nΩ)[rows, :],
             pn.Az[rows, :] - Ω[3] * I(nΩ)[rows, :], [1.0 zeros(1, nΩ - 1)])
    rhs = [zeros(3 * length(rows)); 1 / sqrt(4π)]
    Y = M \ rhs
    @assert norm(M * Y - rhs) < 1e-10 "spherical harmonics recursion not satisfied"
    for l = 0:N # addition theorem Σₘ Y_ℓm² = (2ℓ+1)/(4π)
        @assert abs(sum(Y[l^2+1:(l+1)^2] .^ 2) - (2l + 1) / (4π)) < 1e-8
    end
    return Y
end

degreeOfMoment(i::Int) = floor(Int, sqrt(i - 1)) # i is the 1-based moment index

# uncollided beam

# Straight-ahead CSDA of the uncollided beam along one row of interior cells with materials mats[i] and densities ρ[i] -> currently uncollided can only compute beams aligned with the grid!
function uncollidedRow(s::Settings, beam::ProtonBeam, num::ProtonNumerics, mats::Vector{ProtonMaterial}, ρ::Vector{Float64}, Egrid)
    nx = s.NCells; Δx = s.Δx
    σE = beam.σE_rel * beam.E0
    E0s = collect(range(beam.E0 - 5σE, beam.E0 + 5σE, length=num.nSamples))
    f = exp.(-(E0s .- beam.E0) .^ 2 ./ (2σE^2)) ./ (sqrt(2π) * σE)
    Σfac = 0.5 * num.Nc * (num.Nc + 1)
    Ecell = fill(-Inf, nx, num.nSamples)   # energy at cell centres (-Inf: stopped)
    Acell = zeros(nx, num.nSamples)        # attenuation of the uncollided beam
    E = Base.copy(E0s); logA = zeros(num.nSamples); alive = trues(num.nSamples)
    for i = 2:nx-1
        m = mats[i]
        Rcut = residualRange(m, num.Ecut)
        for half = 1:2
            for k in findall(alive)
                Rnew = residualRange(m, E[k]) - ρ[i] * 0.5Δx
                if Rnew <= Rcut
                    alive[k] = false; continue
                end
                Enew = energyAtRange(m, Rnew)
                logA[k] -= Σfac * (attenuationIntegral(m, E[k]) - attenuationIntegral(m, Enew))
                E[k] = Enew
            end
            if half == 1
                Ecell[i, alive] .= E[alive]; Acell[i, alive] .= exp.(logA[alive])
            end
        end
        any(alive) || break
    end
    prof = zeros(nx, length(Egrid))
    for i = 2:nx-1
        valid = findall(Ecell[i, :] .> -Inf)
        length(valid) < 2 && continue
        Ev = Ecell[i, valid]; E0v = E0s[valid]; Av = Acell[i, valid]; fv = f[valid]
        for (n, En) in enumerate(Egrid)
            (En < Ev[1] || En > Ev[end]) && continue
            j = min(searchsortedlast(Ev, En), length(Ev) - 1)
            t = (En - Ev[j]) / (Ev[j+1] - Ev[j])
            dE0dE = (E0v[j+1] - E0v[j]) / (Ev[j+1] - Ev[j])
            prof[i, n] = ((1 - t) * Av[j] * fv[j] + t * Av[j+1] * fv[j+1]) * dE0dE * ρ[i] * stoppingPower(mats[i], En)
        end
    end
    return prof
end


# problem set-up
function ProtonProblem(s::Settings, beam::ProtonBeam, phantom::Phantom, unc::ProtonUncertainty=ProtonUncertainty(),
                       num::ProtonNumerics=ProtonNumerics())
    nx = s.NCells; ny = nx
    num.Nc <= 0 && (num.Nc = s.nPN)
    mats = phantom.materials
    nMat = length(mats)
    σm = [get(unc.σSPR, m.name, 0.0) for m in mats]
    ξ, pξ = parameterQuadrature(s.Nxi, unc.distξ)
    η, pη = parameterQuadrature(s.Neta, unc.distη)
    fm = [1.0 .+ σm[m] .* ξ for m in 1:nMat]   # density factors per material at the ξ-nodes
    @assert all(all(f .> 0) for f in fm) "density must stay positive at all quadrature nodes"
    interior = interiorMask(s)

    # energy grid, dE limited by the transport CFL with the smallest stopping power (at E_max)
    Emax = beam.E0 + 4 * beam.σE_rel * beam.E0
    Smin = minimum(phantom.ρ0[c] * minimum(fm[phantom.matIndex[c]]) * stoppingPower(mats[phantom.matIndex[c]], Emax)
                   for c in findall(interior .> 0))
    nE = Int(ceil((Emax - num.Ecut) / (num.cflE * s.Δx * Smin))) + 1
    Egrid = collect(range(Emax, num.Ecut, length=nE))
    dE = Egrid[1] - Egrid[2]
    s.Δt = dE

    # P_N system and angular operators
    pn = PNSystem(s); SetupSystemMatrices(pn)
    nΩ = pn.nTotalEntries
    roe(A) = (V = eigvecs(A); V * Diagonal(abs.(eigvals(A))) * inv(V))
    Y_b = sphericalHarmonicsAt(pn, [1.0, 0.0, 0.0])   # beam direction +x
    ℓ = [degreeOfMoment(i) for i in 1:nΩ]
    aΩ = 0.5 .* (num.Nc * (num.Nc + 1) .- ℓ .* (ℓ .+ 1)) .* Y_b
    LFP = Diagonal(0.5 .* ℓ .* (ℓ .+ 1))

    # collided operator
    stencil = Stencil(s, nx, ny)
    Iξ = Diagonal(ones(s.Nxi)); Iη = Diagonal(ones(s.Neta))
    terms = Any[]
    for m in 1:nMat
        χm = Float64.(phantom.matIndex .== m)
        any(χm .> 0) || continue
        for (D, A) in ((stencil.Dx, pn.Ax), (stencil.Dxx, roe(pn.Ax)), (stencil.Dy, pn.Az), (stencil.Dyy, roe(pn.Az)))
            push!(terms, (-D * Diagonal(χm ./ phantom.ρ0), A, Diagonal(1.0 ./ fm[m]), Iη, E -> 1.0 / stoppingPower(mats[m], E)))
        end
        push!(terms, (-Diagonal(χm), LFP, Iξ, Iη, E -> transportCoefficient(mats[m], E) / stoppingPower(mats[m], E)))
    end

    # uncollided beam, lateral profile and one CSDA trace per distinct row (materials and densities along x)
    lateral = [exp(-(s.xMid[j] - (beam.y0 + unc.σSetup * η[l]))^2 / (2beam.w^2)) / (sqrt(2π) * beam.w) for j = 1:ny, l = 1:s.Neta]
    reached = vec(maximum(lateral, dims=2)) .> 1e-12 * maximum(lateral)
    rowClass = zeros(Int, ny); signatures = Dict{Any,Int}(); rows = Int[]
    for j = 2:ny-1
        reached[j] || continue
        cells = [vectorIndex(ny, i, j) for i = 1:nx]
        key = (phantom.matIndex[cells[2:nx-1]], phantom.ρ0[cells[2:nx-1]])
        rowClass[j] = get!(signatures, key) do
            push!(rows, j); length(rows)
        end
    end
    profiles = zeros(nx, nE, length(rows), s.Nxi)
    for (c, j) in enumerate(rows), k = 1:s.Nxi
        cells = [vectorIndex(ny, i, j) for i = 1:nx]
        matRow = mats[phantom.matIndex[cells]]
        ρRow = phantom.ρ0[cells] .* [fm[phantom.matIndex[cl]][k] for cl in cells]
        profiles[:, :, c, k] .= uncollidedRow(s, beam, num, matRow, ρRow, Egrid)
    end

    return ProtonProblem(beam, phantom, unc, num, σm, ξ, pξ, η, pη, Egrid, dE, aΩ, terms, profiles, rowClass, lateral)
end

# fields per energy step

# uncollided ψ̃_u (= its scalar flux φ̃_u, since the beam is mono-directional) at grid energy n
function uncollidedField(s::Settings, prob::ProtonProblem, n::Int)
    nx = s.NCells; ny = nx
    F = zeros(nx * ny, s.Nxi, s.Neta)
    for j = 2:ny-1
        c = prob.rowClass[j]
        c == 0 && continue
        for k = 1:s.Nxi, l = 1:s.Neta, i = 2:nx-1
            F[vectorIndex(ny, i, j), k, l] = prob.profiles[i, n, c, k] * prob.lateral[j, l]
        end
    end
    return F
end

# spatial factor of the first-collision source
function sourceField(s::Settings, prob::ProtonProblem, n::Int, φ̃u=uncollidedField(s, prob, n))
    E = prob.Egrid[n]
    ratio = [transportCoefficient(m, E) / stoppingPower(m, E) for m in prob.phantom.materials]
    return ratio[prob.phantom.matIndex] .* φ̃u
end

# density 
protonDensity(prob::ProtonProblem) = prob.phantom.ρ0 .* (1.0 .+ prob.σm[prob.phantom.matIndex] .* prob.ξ')

# set the energy-dependent operator coefficients in the Rhs
function setEnergy!(rhs::Rhs, prob::ProtonProblem, E::Float64)
    for (k, t) in enumerate(prob.terms)
        rhs.A[k][1] = t[5](E) .* t[1]
    end
end

# replace the source TTN of the Rhs in place
function setSource!(rhs::Rhs, Q::TTN)
    rhs.QRhs.C = Q.C
    rhs.QRhs.leaves = Q.leaves
    rhs.QRhs.S = Q.S
    rhs.QRhs.VᵀFV = Q.VᵀFV
end

# solvers

# TTN BUG integration in energy; returns dose (ncells × nξ × nη), final TTN and ranks
function SolveProtonTTN(solver, prob::ProtonProblem) # solver::BUGIntegrator (defined later)
    s = solver.settings; rhs = solver.rhs
    n = s.NCells^2
    ρ = protonDensity(prob)
    Y = generateRadTree2D(s)
    nE = length(prob.Egrid)
    dose = zeros(n, s.Nxi, s.Neta)
    rVec = zeros(nE - 1, length(get_rank(Y)))
    φ̃u = uncollidedField(s, prob, 1)
    prog = Progress(nE - 1, 1)
    for m = 1:nE-1
        rVec[m, :] .= get_rank(Y)
        setEnergy!(rhs, prob, prob.Egrid[m])
        setSource!(rhs, generateSourceTTN(s, sourceField(s, prob, m, φ̃u); tol=prob.num.tolSource, angle=prob.aΩ))
        FY = eval(rhs, Y)
        _, Ĉ¹, Û¹ = Step(solver, Y, FY)
        Y = θ(solver, TTN(Y.id, Û¹, Ĉ¹, ones(1, 1), ones(1, 1)))
        φ̃u = uncollidedField(s, prob, m + 1)
        φ̃c = sqrt(4π) .* reshape(eval(Y, Any[:, 1, :, :]), n, s.Nxi, s.Neta)
        for l = 1:s.Neta
            dose[:, :, l] .+= prob.dE .* (φ̃u[:, :, l] .+ φ̃c[:, :, l]) ./ ρ
        end
        next!(prog)
    end
    return dose, Y, rVec
end

# full-tensor reference with identical discretisation (= stochastic collocation on the same nodes)
function SolveProtonFull(s::Settings, rhs::Rhs, prob::ProtonProblem)
    n = s.NCells^2; nΩ = length(prob.aΩ)
    ρ = protonDensity(prob)
    U = zeros(n, nΩ, s.Nxi, s.Neta)
    nE = length(prob.Egrid)
    dose = zeros(n, s.Nxi, s.Neta)
    φ̃u = uncollidedField(s, prob, 1)
    prog = Progress(nE - 1, 1)
    for m = 1:nE-1
        setEnergy!(rhs, prob, prob.Egrid[m])
        q = sourceField(s, prob, m, φ̃u)
        dU = zeros(size(U))
        for Ω = 1:nΩ
            dU[:, Ω, :, :] .= prob.aΩ[Ω] .* q
        end
        for Aₖ in rhs.A
            V = U
            for d = 1:4
                V = modeprod(V, Aₖ[d], d)
            end
            dU .+= V
        end
        U .+= prob.dE .* dU
        φ̃u = uncollidedField(s, prob, m + 1)
        for l = 1:s.Neta
            dose[:, :, l] .+= prob.dE .* (φ̃u[:, :, l] .+ sqrt(4π) .* U[:, 1, :, l]) ./ ρ
        end
        next!(prog)
    end
    return dose
end
