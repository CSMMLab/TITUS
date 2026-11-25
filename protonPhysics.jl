# Proton interaction data 
# - stopping powers: element-wise TOPAS tables (data/proton_S_data_topas), materials are represented by weighted composition
# - elastic (Coulomb) scattering: screened Rutherford cross section with nuclear form factor
# - Fokker-Planck transport coefficient
# - HU to material compositions from the Schneider tables in matComp.
# All quantities are returned per unit density (ρ = 1 g/cm³), i.e. S [MeV cm²/g] and ξ [cm²/g],
# so that the macroscopic values are ρ S and ρ ξ. Energies are kinetic energies in MeV.

using LinearAlgebra
using FastGaussQuadrature

const PROTON_ELEMENTS = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "Ar", "K", "Ca"]
const PROTON_Z = [1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20]
const PROTON_MOLW = [1.008, 12.011, 14.007, 15.999, 22.989, 24.305, 30.973, 32.060, 35.450, 39.948, 39.098, 40.078]

# mass fractions in % (order of PROTON_ELEMENTS)
const COMP_WATER = [11.19, 0, 0, 88.81, 0, 0, 0, 0, 0, 0, 0, 0]
# TITUS matComp, 900 < HU ≤ 1000 (cortical-bone-like, Schneider et al. 2000)
const COMP_BONE = [5.2, 24.6, 3.7, 41.1, 0.1, 0.2, 7.8, 0.3, 0.0, 0.0, 0.0, 17.0]

# read element-wise mass stopping powers 
function readStoppingPowerTable(file=joinpath(@__DIR__, "data", "proton_S_data_topas"))
    open(file, "r") do io
        n = parse(Int, strip(readline(io)))
        E = [parse(Float64, strip(readline(io))) for _ in 1:n]
        S = Dict{String,Vector{Float64}}()
        for _ in 1:12
            el = strip(readline(io))
            S[el] = [parse(Float64, strip(readline(io))) for _ in 1:n]
        end
        return E, S
    end
end

# piecewise linear interpolation in log-log (tables are smooth power laws)
function loglogInterp(xt::Vector{Float64}, yt::Vector{Float64}, x::Real)
    x <= xt[1] && return yt[1]
    x >= xt[end] && return yt[end]
    i = searchsortedlast(xt, x)
    t = (log(x) - log(xt[i])) / (log(xt[i+1]) - log(xt[i]))
    return exp((1 - t) * log(yt[i]) + t * log(yt[i+1]))
end

# mass stopping power of a mixture [MeV cm²/g]
function massStoppingPower(comp::Vector{<:Real}, E::Real, table=readStoppingPowerTable())
    Et, St = table
    return sum(comp[k] / 100 * loglogInterp(Et, St[el], E) for (k, el) in enumerate(PROTON_ELEMENTS) if comp[k] > 0)
end

# elastic differential cross section per unit density of a pure element k [1/(cm sr)],
# screened Rutherford with nuclear and proton form factors
function elasticXSElement(k::Int, E::Real, μ::Real)
    m_p = 938.2720813; m_e = 0.5109989461; N_A = 6.02214076e23; α = 0.0072973525
    ħc = 0.19732697e-10                     # MeV cm (labelled "MeV·m" in TITUS, but the value is in cm)
    g_to_MeV = 1.0 / (1.78266192e-36 * 1.0e9)
    Z = PROTON_Z[k]; A = PROTON_MOLW[k]
    N = N_A / A                             # atoms per gram
    m_t = A / N_A * g_to_MeV - Z * m_e      # nuclear mass in MeV
    com_to_lab = (1 + 2μ * m_p / m_t + (m_p / m_t)^2)^1.5 / (1 + μ * m_p / m_t)
    γ = E / m_p + 1
    pc = m_p * sqrt(γ^2 - 1)
    α² = (Z * α)^2 / (1 - 1 / γ^2)
    invk² = (ħc / pc)^2
    χ₀² = (1.13 * α * Z^(1 / 3) * m_e / pc)^2
    χα² = χ₀² * (1.13 + 3.76 * α²)
    q_screen = 1 / (1 - μ + χα²)^2
    R_inv = 1 / (1.27 * round(A)^(1 / 3) / 197.3269804)
    t = 2 * pc^2 * (1 - μ)
    F_A = exp(-t / R_inv^2)
    F_p = 1 / (1 + t * 1e-6 / 0.71)^2
    F_N = (1 - 1 / Z) * F_A + F_p / Z
    return 4 * N * α² * invk² * q_screen * F_N * com_to_lab          # 1/(cm sr) per g/cm³
end

# Fokker-Planck transport coefficient per unit density
function massTransportCoefficient(comp::Vector{<:Real}, E::Real; titusQuadrature=true)
    if titusQuadrature
        μ, w = gausslegendre(50)
        f = [sum(comp[k] / 100 * elasticXSElement(k, E, μⱼ) for k in 1:12 if comp[k] > 0) * (1 - μⱼ) for μⱼ in μ]
        return 2π * dot(w, f)
    end
    # accurate integration in t = 1-μ on a logarithmic grid: ∫₀² σ(1-t) t dt = ∫ σ(1-t) t² d(log t)
    s = range(log(1e-14), log(2.0), length=4001)
    vals = [sum(comp[k] / 100 * elasticXSElement(k, E, 1 - exp(sⱼ)) for k in 1:12 if comp[k] > 0) * exp(2sⱼ) for sⱼ in s]
    return 2π * step(s) * (sum(vals) - 0.5 * (vals[1] + vals[end]))
end

# tabulated material properties on a fine energy grid, linear interpolation in between
struct ProtonMaterial
    name::String
    comp::Vector{Float64}
    Etab::Vector{Float64}
    S::Vector{Float64}   # mass stopping power [MeV cm²/g]
    ξ::Vector{Float64}   # mass FP transport coefficient [cm²/g]
    ER::Vector{Float64}  # fine energy grid for the range tables
    R::Vector{Float64}   # residual (CSDA) range ∫₀ᴱ dE/S [g/cm²]
    T::Vector{Float64}   # ∫ ξ/S dE from Etab[1] (dimensionless), attenuation integral of the uncollided beam
end
function ProtonMaterial(name, comp; Emin=0.5, Emax=300.0, n=400)
    Etab = exp.(range(log(Emin), log(Emax), length=n))
    table = readStoppingPowerTable()
    S = [massStoppingPower(comp, E, table) for E in Etab]
    ξ = [massTransportCoefficient(comp, E) for E in Etab]
    # cumulative integrals on a 10× finer grid with the trapezoidal rule in log E (integrands E/S and Eξ/S are
    # smooth); the fine grid keeps interpolation kinks out of the Jacobian dE₀/dE of the uncollided beam
    ER = exp.(range(log(Emin), log(Emax), length=10n))
    h = log(ER[2]) - log(ER[1])
    SR = [loglogInterp(Etab, S, E) for E in ER]; ξR = [loglogInterp(Etab, ξ, E) for E in ER]
    fR = ER ./ SR; fT = ER .* ξR ./ SR
    R = zeros(length(ER)); T = zeros(length(ER))
    R[1] = ER[1] / (1.77 * SR[1])  # range below Emin from the power law R ∝ E^1.77
    for i = 2:length(ER)
        R[i] = R[i-1] + 0.5h * (fR[i-1] + fR[i])
        T[i] = T[i-1] + 0.5h * (fT[i-1] + fT[i])
    end
    return ProtonMaterial(name, Float64.(comp), Etab, S, ξ, ER, R, T)
end
stoppingPower(m::ProtonMaterial, E) = loglogInterp(m.Etab, m.S, E)
transportCoefficient(m::ProtonMaterial, E) = loglogInterp(m.Etab, m.ξ, E)
residualRange(m::ProtonMaterial, E) = loglogInterp(m.ER, m.R, E)          # [g/cm²]
energyAtRange(m::ProtonMaterial, R) = loglogInterp(m.R, m.ER, R)           # inverse of residualRange
function attenuationIntegral(m::ProtonMaterial, E)                          # ∫ ξ/S dE, linear in log E
    E <= m.ER[1] && return m.T[1]
    E >= m.ER[end] && return m.T[end]
    i = searchsortedlast(m.ER, E)
    t = (log(E) - log(m.ER[i])) / (log(m.ER[i+1]) - log(m.ER[i]))
    return (1 - t) * m.T[i] + t * m.T[i+1]
end

# CSDA range in a material of density ρ [cm]
function csdaRange(m::ProtonMaterial, E0; ρ=1.0, Emin=m.Etab[1])
    Es = exp.(range(log(Emin), log(E0), length=2000))
    f = [1 / (ρ * stoppingPower(m, E)) for E in Es]
    return sum(0.5 .* (f[1:end-1] .+ f[2:end]) .* diff(Es))
end
