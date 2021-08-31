using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using DelimitedFiles

close("all")

s = Settings(1002,1002);

if s.problem == "AirCavity"
    smapIn = readdlm("dose_ac.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
elseif s.problem == "WaterPhantomKerstin"
    smapIn = readdlm("doseStarmapWaterPhantom.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
else
    xRef = 0; doseRef = 1;
end

############################
solver = SolverCSD(s)

@time u, dose = Solve(solver);

#fig = figure("Dose Contour",figsize=(10,10))
#ax = fig.add_subplot(2,1,1,projection="3d")
#plot_surface(s.xMid,s.yMid,dose, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
#xlabel("X")
#ylabel("Y")
#PyPlot.title("Surface Plot")

#subplot(212)
#ax = fig.add_subplot(2,1,2)
#cp = contour(s.xMid,s.yMid,dose, colors="black", linewidth=2.0)
#ax.clabel(cp, inline=1, fontsize=10)
#xlabel("X")
#ylabel("Y")
#PyPlot.title("Contour Plot")
#tight_layout()

#surf(s.xMid,s.yMid,dose,st=:surface,camera=(-30,30));
#fig = figure("pyplot_surfaceplot",figsize=(10,10))
#plot_surface(s.xMid,s.yMid,dose, rstride=2, cstride=2, cmap=ColorMap("viridis"), alpha=0.8)

#surf(s.xMid,s.yMid,dose,st=:surface,camera=(-30,30));
#fig = figure("u0",figsize=(10,10))
#plot_surface(s.xMid,s.yMid,u[:,:,1], rstride=2, cstride=2, cmap=ColorMap("viridis"), alpha=0.8)

fig = figure("Dose Contour",figsize=(10,10),dpi=100)

pcolormesh(dose)
#colorbar()

fig = figure("u Contour",figsize=(10,10),dpi=100)

pcolormesh(u[:,:,1])

println("main finished")
