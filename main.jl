using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("settings.jl")
include("TTN.jl")
include("Rhs.jl")
include("Problem.jl")
include("BUGIntegrator.jl")
include("ParallelIntegratorInefficient.jl")
include("ParallelIntegrator.jl")
using PyCall
using PyPlot
using ProgressMeter
np = pyimport("numpy")

close("all")

s = Settings(101, 21, 100, 100, "radiation2DUQ")
s.nPN = 31
s.ϵ = 2*1e-2;
solver = ParallelIntegrator(s)
YPar, rParallel = Solve(solver)

s.ϵ = 3*1e-2;
solver = BUGIntegrator(s)
YBUG, rBUG = Solve(solver)

n = s.NCells^2
nξ = s.Nxi
nη = s.Neta
r = s.r

ξ, wξ = gausslegendre(nξ);
η, wη = gausslegendre(nη);

# compute moments
ρ = reshape(eval(YPar, [:, 1,:,:]), (n, nξ, nη));
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

rho = Vec2Mat(s.NCells,s.NCells,Eρ);
rhoBUG = Vec2Mat(s.NCells,s.NCells,EρBUG);
rhoVar = Vec2Mat(s.NCells,s.NCells,σ²);
rhoVarBUG = Vec2Mat(s.NCells,s.NCells,σ²BUG);


X = (s.xMid[2:end-1]'.*ones(size(s.xMid[2:end-1])));
Y = (s.xMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))';

## Expected value
fig = figure("parallel",figsize=(10,10),dpi=100)
ax = gca()
maxsol = maximum(4.0*pi*sqrt(2)*rho[2:(end-1),(end-1):-1:2]')
pcolormesh(X,Y,4.0*pi*sqrt(2)*rho[2:(end-1),(end-1):-1:2]',vmin=0,vmax=maxsol)
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\mathbb{E}[\Phi]$, parallel BUG", fontsize=25)
tight_layout()
show()
savefig("scalar_flux_PN_$(s.problem)_nx$(s.NCells)_N$(s.nPN)_parallel.png")

fig = figure("augmented BUG",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoBUG[2:(end-1),(end-1):-1:2]',vmin=0,vmax=maxsol)
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\mathbb{E}[\Phi]$, augmented BUG", fontsize=25)
tight_layout()
show()
savefig("scalar_flux_PN_$(s.problem)_nx$(s.NCells)_N$(s.nPN)_augmented.png")

## Variance
fig = figure("parallel var",figsize=(10,10),dpi=100)
ax = gca()
maxsol = maximum(4.0*pi*sqrt(2)*rhoVar[2:(end-1),(end-1):-1:2]')
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoVar[2:(end-1),(end-1):-1:2]',vmin=0,vmax=maxsol)
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"Var$[\Phi]$, parallel BUG", fontsize=25)
tight_layout()
show()
savefig("scalar_flux_var_PN_$(s.problem)_nx$(s.NCells)_N$(s.nPN)_augmented.png")

fig = figure("augmented BUG var",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(X,Y,4.0*pi*sqrt(2)*rhoVarBUG[2:(end-1),(end-1):-1:2]',vmin=0,vmax=maxsol)
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"Var$[\Phi]$, augmented BUG", fontsize=25)
tight_layout()
show()
savefig("scalar_flux_var_PN_$(s.problem)_nx$(s.NCells)_N$(s.nPN)_parallel.png")

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
ax.set_ylim([0,250])
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("t", fontsize=15)
ax.set_ylabel("rank, BUG", fontsize=15)
show()
tight_layout()
savefig("ranksBUG2D.png")

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
ax.set_ylim([0,250])
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("t", fontsize=15)
ax.set_ylabel("rank, parallel", fontsize=15)
show()
tight_layout()
savefig("ranksParallel2D.png")
