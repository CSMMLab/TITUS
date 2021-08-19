include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
# pygui(:agg)
# ENV["MPLBACKEND"]="agg"
# using Plots
# Plots.PyPlotBackend()
using DelimitedFiles

close("all")

s = Settings(302);

############################
solver = SolverCSD(s)

@time u, dose = Solve(solver);

fig, ax = subplots()
ax[:plot](s.xMid,u[:,1], "r--", linewidth=2, label="PN", alpha=0.6)
ax[:legend](loc="upper left")
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=20) 
fig.savefig("scalarFlux.png", dpi=fig.dpi)

fig, ax = subplots()
ax[:plot](s.xMid,dose./maximum(dose), "r--", linewidth=2, label="PN dose", alpha=0.6)
ax[:legend](loc="upper left")
ax.set_xlim([s.a,s.b])
ax.set_ylim([0,1])
ax.tick_params("both",labelsize=20) 
fig.savefig("dose.png", dpi=fig.dpi)

println("main finished")
