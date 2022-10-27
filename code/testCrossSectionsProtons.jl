include("utils.jl")
include("settings.jl")
include("CSD.jl")
include("MaterialParameters.jl")
include("MaterialParametersProtons.jl")

using PyPlot
using DelimitedFiles
using LegendrePolynomials

s = Settings();
s.particle = "Protons"

# plot proton cross sections

csd = CSD(s);
param = MaterialParametersProtons();

# plot proton cross sections
E = csd.eGrid;#[10.0 25.0 50.0] .+ s.eRest
nE = length(energy)
xi = zeros(nE, s.nPN + 1)
for n = 1:nE
    xi[n,:] = SigmaAtEnergy(csd, E[n])
end

# plot stopping power
fig = figure("stopping power",figsize=(14,12),dpi=100)
ax = gca()
ax.plot(csd.eGrid[1:end-1] .- s.eRest,csd.S[1:end-1], linewidth=2, alpha=0.6)
ax.set_xscale("log")
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("E (MeV)",fontsize=15)
ax.set_ylabel(L"S",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
show()

# plot sigma
fig = figure("sigma",figsize=(14,12),dpi=100)
ax = gca()
idx = [1;2;3;4;5]
lt = ["k-","r-","g-","b-","m-","y-"]
ltS = ["k:","r:","g:","b:","m:","y:"]
for l = 1:length(idx)
    ax.loglog(E,xi[:,idx[l]], lt[l], linewidth=2, label="order $(idx[l])", alpha=0.6)
end
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("E (MeV)",fontsize=15)
ax.set_ylabel(L"\sigma",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
show()
