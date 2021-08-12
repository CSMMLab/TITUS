include("settings.jl")
include("Solver.jl")

using PyCall
using PyPlot
# pygui(:agg)
# ENV["MPLBACKEND"]="agg"
# using Plots
# Plots.PyPlotBackend()
using DelimitedFiles

s = Settings(1002);

############################
solver = Solver(s)

@time tEnd, u = Solve(solver);


s.tEnd = tEnd;

v = readdlm("/home/pia/CSD-DLRA/code/PlaneSourceRaw", ',')
#v = readdlm("PlaneSourceRaw", ',')
uEx = zeros(length(v));
for i = 1:length(v)
    if v[i] == ""
        uEx[i] = 0.0;
    else
        uEx[i] = Float64(v[i])
    end
end
x = collect(range(-1.5,stop=1.5,length=(2*length(v)-1)));
uEx = [uEx[end:-1:2];uEx]


fig, ax = subplots()
ax[:plot](x,uEx, "k-", linewidth=2, label="exact", alpha=0.6)
ax[:plot](s.xMid,u[:,1], "r--", linewidth=2, label="PN", alpha=0.6)
ax[:legend](loc="upper left")
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=20) 
fig.savefig("output.png", dpi=fig.dpi)


println("main finished")
