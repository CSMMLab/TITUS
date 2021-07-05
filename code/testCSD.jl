include("settings.jl")
include("Solver.jl")
include("CSD.jl")
include("MaterialParameters.jl")

using PyPlot

s = Settings(100);

csd = CSD(s);
param = MaterialParameters();

fig, ax = subplots()
ax.plot(param.E_tab,param.S_tab, "k-", linewidth=2, label="S_tab", alpha=0.6)
ax.plot(csd.eGrid,csd.S, "r--o", linewidth=2, label="S", alpha=0.6)
ax.legend(loc="upper left")
ax.set_xlim([csd.eGrid[1],csd.eGrid[end]])
ax.tick_params("both",labelsize=20) 
show()
