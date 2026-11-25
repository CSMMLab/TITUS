# Proton pencil beam through a water phantom with bone insert, Gaussian stopping-power-ratio and lateral setup error
# Solver is simplified 1st collision source (raytracer + DLRA) in 2D, collided flux is computed with the TTN BUG integrator.
#
# usage: julia --project=. mainProtonUQ.jl [NCells] [nPN] [nξ=nη] [ϵ] [E0 in MeV] [runReference 0/1]
ENV["MPLBACKEND"] = "Agg"
using Pkg
Pkg.activate(@__DIR__)

baremodule ProtonRun # need this to run on Julia ≥ 1.12
using Base
const _included = Set{String}()
function include(p)
    basename(p) in _included && return nothing
    push!(_included, basename(p))
    return Base.include(ProtonRun, p)
end

using ProgressMeter
using DelimitedFiles
using NPZ
include("settings.jl")
include("Rhs.jl")
include("TTN.jl")
include("BUGIntegrator.jl")
using PyCall
using PyPlot
np = pyimport("numpy")

NCells = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
nPN    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 11
nq     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
ϵ      = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-2
E0     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 90.0
runRef = length(ARGS) >= 6 ? parse(Int, ARGS[6]) == 1 : false

s = Settings(NCells + 1, nPN, nq, nq, "ProtonUQ")
s.ϵ = ϵ
s.r = min(s.r, nq, GlobalIndex(s.nPN, s.nPN) + 1)

# ---------------- case definition (beam, geometry, uncertainties, numerics) -----------------
beam = ProtonBeam(E0=E0, y0=4.0, w=0.5)
phantom = waterPhantomWithInsert(s; insX=[1.5, 3.0], insY=[4.0, 6.5])
unc = ProtonUncertainty(σSPR=Dict("water" => 0.016, "bone" => 0.024), σSetup=0.3)
num = ProtonNumerics()

tSetup = @elapsed prob = ProtonProblem(s, beam, phantom, unc, num)
s.proton = prob
solver = BUGIntegrator(s)
mats = phantom.materials

n = s.NCells^2
nΩ = GlobalIndex(s.nPN, s.nPN) + 1
println("ProtonUQ: E0 = $E0 MeV, $(s.NCells)² cells (Δx = $(round(10*s.Δx, digits=2)) mm), P$(s.nPN) ($nΩ moments), ",
        "$(s.Nxi)×$(s.Neta) nodes, ϵ = $ϵ, $(length(prob.Egrid)) energies, dE = $(round(prob.dE, digits=3)) MeV, setup $(round(tSetup, digits=1)) s")
for (k, m) in enumerate(mats)   # relative stopping power w.r.t. the first material, at E0 and nominal density
    ρm = phantom.ρ0[findfirst(phantom.matIndex .== k)]
    println("material $(m.name): ρ₀ = $ρm g/cm³, SPR σ = $(prob.σm[k]), CSDA range at ρ₀ $(round(csdaRange(m, E0; ρ=ρm), digits=2)) cm, ",
            "RSP rel. to $(mats[1].name) $(round(ρm * stoppingPower(m, E0) / stoppingPower(mats[1], E0), digits=3))")
end
println("uncollided beam traced for $(size(prob.profiles, 3)) distinct rows")

outdir = joinpath(@__DIR__, "results_ProtonUQ"); mkpath(outdir)
tag = "E$(Int(round(E0)))_nx$(s.NCells)_N$(s.nPN)_nq$(nq)_eps$(ϵ)"

tTTN = @elapsed dose, YBUG, rBUG = SolveProtonTTN(solver, prob)
ttnStorage(Y::TTN) = length(Y.C) + sum(ttnStorage, Y.leaves; init=0)
println("TTN BUG run time $(round(tTTN, digits=1)) s, final ranks ", get_rank(YBUG), ", max ranks ", maximum(rBUG, dims=1))
println("TTN storage $(ttnStorage(YBUG)) vs full tensor $(n*nΩ*s.Nxi*s.Neta) entries")

ED, VarD = uqStatistics(dose, prob.pξ, prob.pη)
Dmax = maximum(ED)

if runRef
    tRef = @elapsed doseRef = SolveProtonFull(s, solver.rhs, prob)
    EDRef, VarDRef = uqStatistics(doseRef, prob.pξ, prob.pη)
    errE = norm(ED - EDRef) / norm(EDRef)
    errStd = norm(sqrt.(max.(VarD, 0)) - sqrt.(max.(VarDRef, 0))) / norm(sqrt.(VarDRef))
    errAll = norm(dose - doseRef) / norm(doseRef)
    println("full reference run time $(round(tRef, digits=1)) s; relative errors TTN vs reference: E[D] $errE, Std[D] $errStd, all nodes $errAll")
    writedlm(joinpath(outdir, "errors_$tag.txt"), ["errE" errE; "errStd" errStd; "errAllNodes" errAll; "tTTN" tTTN; "tRef" tRef])
end

# laterally integrated depth dose (IDD) per parameter node and its statistics
iddNodes = zeros(s.NCells, s.Nxi, s.Neta)
for i = 1:s.NCells, j = 1:s.NCells
    iddNodes[i, :, :] .+= s.Δx .* dose[vectorIndex(s.NCells, i, j), :, :]
end
EIDD, VarIDD = uqStatistics(iddNodes, prob.pξ, prob.pη)

npzwrite(joinpath(outdir, "data_$tag.npz"), Dict(
    "xMid" => s.xMid, "ED" => ED, "VarD" => VarD, "ranks" => rBUG, "Egrid" => prob.Egrid,
    "EIDD" => EIDD, "StdIDD" => sqrt.(max.(VarIDD, 0)), "E0" => [E0], "matIndex" => phantom.matIndex,
    "rho0" => phantom.ρ0, "y0" => [beam.y0], "w" => [beam.w], "NCells" => [s.NCells],
    "EDRef" => runRef ? EDRef : zeros(0), "VarDRef" => runRef ? VarDRef : zeros(0)))

# ---------------- plots --------------------------------------------------------------------
xm = s.xMid; inn = 2:s.NCells-1
toMat(v) = Vec2Mat(s.NCells, s.NCells, v)[inn, inn]' # rows: y, columns: x
matMap = Vec2Mat(s.NCells, s.NCells, Float64.(phantom.matIndex))   # [i (x), j (y)]
bg = argmax([count(==(k), phantom.matIndex) for k in eachindex(mats)])   # background material
# contiguous coordinate ranges [a, b] where mask is true
function spans(mask, c)
    idx = findall(mask); isempty(idx) && return Tuple{Float64,Float64}[]
    br = [0; findall(diff(idx) .> 1); length(idx)]
    return [(c[idx[br[q]+1]] - 0.5s.Δx, c[idx[br[q+1]]] + 0.5s.Δx) for q in 1:length(br)-1]
end
function decorate(ax)
    length(mats) > 1 && ax.contour(xm[inn], xm[inn], matMap[inn, inn]', levels=collect(1.5:1:length(mats)),
                                   colors="w", linewidths=1.2, linestyles="--")
    #ax.annotate("", xy=(0.8, beam.y0), xytext=(0.0, beam.y0), arrowprops=Dict("arrowstyle"=>"->", "color"=>"w", "lw"=>2))
    ax.set_xlabel("depth x [cm]"); ax.set_ylabel("y [cm]"); ax.set_aspect("equal")
end
fig, axs = subplots(1, 2, figsize=(11, 4.6), dpi=150)
im = axs[1].pcolormesh(xm[inn], xm[inn], 100 .* toMat(ED) ./ Dmax, shading="auto", cmap="turbo", vmin=0, vmax=100)
colorbar(im, ax=axs[1], label="% of max"); axs[1].set_title(L"$\mathbb{E}[D]$")
im = axs[2].pcolormesh(xm[inn], xm[inn], 100 .* toMat(sqrt.(max.(VarD, 0))) ./ Dmax, shading="auto", cmap="viridis")
colorbar(im, ax=axs[2], label=L"% of max $\mathbb{E}[D]$"); axs[2].set_title(L"Std$[D]$")
for ax in axs; decorate(ax); end
tight_layout()
savefig(joinpath(outdir, "protonUQ_dose_$tag.png")); savefig(joinpath(outdir, "protonUQ_dose_$tag.pdf"))

# depth-dose curves at y0 ± w (labelled with the materials they cross, non-background materials shaded),
# lateral profile at the depth of the dose maximum
rowLabel(j) = "y = $(round(xm[j], digits=2)) (" * join(unique(m.name for m in mats[Int.(matMap[inn, j])]), ", ") * ")"
dep(v, j) = 100 .* Vec2Mat(s.NCells, s.NCells, v)[inn, j] ./ Dmax
iPk = inn[argmax(vec(maximum(Vec2Mat(s.NCells, s.NCells, ED)[inn, inn], dims=2)))]
lat(v) = 100 .* Vec2Mat(s.NCells, s.NCells, v)[iPk, inn] ./ Dmax
fig, axs = subplots(1, 2, figsize=(11, 4.0), dpi=150)
for (j, c) in ((argmin(abs.(xm .- (beam.y0 - beam.w))), "C0"), (argmin(abs.(xm .- (beam.y0 + beam.w))), "C3"))
    m = dep(ED, j); sd = dep(sqrt.(max.(VarD, 0)), j)
    for (a, b) in spans(matMap[inn, j] .!= bg, xm[inn]); axs[1].axvspan(a, b, color=c, alpha=0.08, zorder=0); end
    axs[1].fill_between(xm[inn], m .- sd, m .+ sd, color=c, alpha=0.25)
    axs[1].plot(xm[inn], m, c * "-", lw=2, label=rowLabel(j))
    runRef && axs[1].plot(xm[inn], dep(EDRef, j), "k--", lw=1)
end
axs[1].set_xlabel("depth x [cm]"); axs[1].set_ylabel(L"dose [% of max $\mathbb{E}[D]$]"); axs[1].legend(fontsize=8)
axs[1].set_title(L"depth dose, $\mathbb{E}\pm$Std" * (runRef ? " (dashed: collocation)" : ""), fontsize=10)
m = lat(ED); sd = lat(sqrt.(max.(VarD, 0)))
for (a, b) in spans(matMap[iPk, inn] .!= bg, xm[inn]); axs[2].axvspan(a, b, color="0.85", zorder=0); end
axs[2].fill_between(xm[inn], m .- sd, m .+ sd, color="C0", alpha=0.25)
axs[2].plot(xm[inn], m, "C0-", lw=2, label=L"$\mathbb{E}[D]$")
runRef && axs[2].plot(xm[inn], lat(EDRef), "k--", lw=1, label="collocation")
axs[2].set_xlabel("y [cm]"); axs[2].set_title("lateral profile at depth x = $(round(xm[iPk], digits=2)) cm", fontsize=10)
axs[2].legend(fontsize=8)
tight_layout()
savefig(joinpath(outdir, "protonUQ_profiles_$tag.png")); savefig(joinpath(outdir, "protonUQ_profiles_$tag.pdf"))

fig, ax = subplots(figsize=(5.5, 4.2), dpi=150)
labels = [L"\tau_1", "space", "angle", L"\tau_2", L"\xi", L"\eta"]
for (c, lab, ls) in zip(2:7, labels, ["-", "-", "-", "--", "-.", ":"])
    ax.plot(prob.Egrid[1:end-1], rBUG[:, c], ls, lw=2, label=lab)
end
ax.invert_xaxis(); ax.set_xlabel("energy E [MeV]"); ax.set_ylabel("rank"); ax.legend(fontsize=9)
tight_layout()
savefig(joinpath(outdir, "protonUQ_ranks_$tag.png")); savefig(joinpath(outdir, "protonUQ_ranks_$tag.pdf"))
println("results written to $outdir")

end # baremodule ProtonRun
