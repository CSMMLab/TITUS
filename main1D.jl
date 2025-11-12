using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ProgressMeter
include("settings.jl")
include("Rhs.jl")
include("Problem.jl")
include("TTN.jl")
include("BUGIntegrator.jl")
include("ParallelIntegrator.jl")
using PyCall
using PyPlot
np = pyimport("numpy")
using DelimitedFiles

close("all")

# define if collocation is rerun
write = false
read = false

s = Settings(1000)
s.problem = "radiationUQ"

solver = BUGIntegrator(s)
YBUG, rBUG = Solve(solver)

n = s.NCells
nξ = s.Nxi
nη = s.Neta
r = s.r

ξ, wξ = gausslegendre(nξ);
η, wη = gausslegendre(nη);

EρCol = readdlm("rhoECollocation.txt", ',')
σ²Col = readdlm("rhoSigCollocation.txt", ',')


# compute moments
ρ = reshape(eval(Y, [:, 1,:,:]), (n, nξ, nη))
ρBUG = reshape(eval(YBUG, [:, 1,:,:]), (n, nξ, nη));
Eρ = zeros(n);
EρBUG = zeros(n);
for j in 1:n
    Eρ[j] += sum(0.25 * wξ * wη' .* ρ[j,:,:])
    EρBUG[j] += sum(0.25 * wξ * wη' .* ρBUG[j,:,:])
end
σ² = zeros(n);
σ²BUG = zeros(n);
for j in 1:n
    σ²[j] += sum(0.25 * wξ * wη' .* (ρ[j,:,:] .- Eρ[j]).^2)
    σ²BUG[j] += sum(0.25 * wξ * wη' .* (ρBUG[j,:,:] .- EρBUG[j]).^2)
end

fig, ax = subplots(figsize=(7.5,5),dpi=100)
ax.plot(s.xMid,Eρ', "b-.", linewidth=2, label=L"$\mathbb{E}[\rho]$, Parallel", alpha=0.6)
ax.plot(s.xMid,EρBUG', "r--", linewidth=2, label=L"$\mathbb{E}[\rho]$, BUG", alpha=0.6)
ax.plot(s.xMid,EρCol, "k-", linewidth=2, label=L"$\mathbb{E}[\rho]$, collocation", alpha=0.3)
ax.legend(loc="upper left",fontsize=15)
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("x", fontsize=15)
show()
tight_layout()
savefig("Erho.png")

fig, ax = subplots(figsize=(7.5,5),dpi=100)
ax.plot(s.xMid,σ²', "b-.", linewidth=2, label=L"$Var[\rho]$, Parallel", alpha=0.6)
ax.plot(s.xMid,σ²BUG', "r--", linewidth=2, label=L"$Var[\rho]$, BUG", alpha=0.6)
ax.plot(s.xMid,σ²Col, "k-", linewidth=2, label=L"$Var[\rho]$, collocation", alpha=0.3)
ax.legend(loc="upper left",fontsize=15)
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("x", fontsize=15)
show()
tight_layout()
savefig("VarRho.png")

Δt = s.Δt
nt = Int(floor(s.tEnd / Δt))
t = collect(range(1, nt)) .* Δt
fig, ax = subplots(figsize=(5.5,5),dpi=100)
ax.plot(t,rBUG[:, 2], "r--", linewidth=2, label=L"\tau_1", alpha=0.7)
ax.plot(t,rBUG[:, 5], "k-", linewidth=2, label=L"\tau_2", alpha=0.5)
ax.plot(t,rBUG[:, 3], "b-.", linewidth=2, label="space", alpha=0.6)
ax.plot(t,rBUG[:, 4], "m:", linewidth=2, label="angle", alpha=0.6)
ax.plot(t,rBUG[:, 6], "g--", linewidth=2, label=L"\xi", alpha=0.6)
ax.plot(t,rBUG[:, 7], "y-", linewidth=2, label=L"\eta", alpha=0.9)
ax.legend(loc="upper left",fontsize=15)
ax.set_xlim([0,s.tEnd])
ax.set_ylim([0,45])
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("t", fontsize=15)
ax.set_ylabel("rank, BUG", fontsize=15)
show()
tight_layout()
savefig("ranksBUG1D.png")

Δt = s.Δt
nt = Int(floor(s.tEnd / Δt))
t = collect(range(1, nt)) .* Δt
fig, ax = subplots(figsize=(5.5,5),dpi=100)
ax.plot(t,rParallel[:, 2], "r--", linewidth=2, label=L"\tau_1", alpha=0.7)
ax.plot(t,rParallel[:, 5], "k-", linewidth=2, label=L"\tau_2", alpha=0.5)
ax.plot(t,rParallel[:, 3], "b-.", linewidth=2, label="space", alpha=0.6)
ax.plot(t,rParallel[:, 4], "m:", linewidth=2, label="angle", alpha=0.6)
ax.plot(t,rParallel[:, 6], "g--", linewidth=2, label=L"\xi", alpha=0.6)
ax.plot(t,rParallel[:, 7], "y-", linewidth=2, label=L"\eta", alpha=0.9)
#ax.legend(loc="upper left",fontsize=15)
ax.set_xlim([0,s.tEnd])
ax.set_ylim([0,45])
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("t", fontsize=15)
ax.set_ylabel("rank, parallel", fontsize=15)
show()
tight_layout()
savefig("ranksParallel1D.png")