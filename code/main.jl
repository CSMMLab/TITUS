using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using DelimitedFiles
using WriteVTK

close("all")

s = Settings(501,501);

if s.problem == "AirCavity"
    smapIn = readdlm("dose_ac.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
elseif s.problem == "WaterPhantomKerstin"
    smapIn = readdlm("doseStarmapWaterPhantom.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
elseif s.problem == "2D"
    doseRef = readdlm("validationData/dose_starmap_full301.txt", Float64)
    xRef = readdlm("validationData/x_starmap_nx301.txt", Float64)
    yRef = readdlm("validationData/y_starmap_ny301.txt", Float64)
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
savefig("output/doseNx$(s.Nx)")

fig = figure("u Contour",figsize=(10,10),dpi=100)

pcolormesh(u[:,:,1])
savefig("output/u0Nx$(s.Nx)")

# line plot dose
fig, ax = subplots()
nyRef = length(yRef)
ax.plot(s.xMid,dose[:,Int(floor(s.NCellsY/2))]./maximum(dose[:,Int(floor(s.NCellsY/2))]), "r--", linewidth=2, label="CSD", alpha=0.8)
ax.plot(xRef',doseRef[:,Int(floor(nyRef/2))]./maximum(doseRef[:,Int(floor(nyRef/2))]), "k-", linewidth=2, label="Starmap", alpha=0.6)
#ax.plot(csd.eGrid,csd.S, "r--o", linewidth=2, label="S", alpha=0.6)
ax.legend(loc="upper left")
ax.set_xlim([s.c,s.d])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
savefig("output/DoseCutYNx$(s.Nx)")

# write vtk file
vtkfile = vtk_grid("output/dose_csd_nx$(s.NCellsX)ny$(s.NCellsY)", s.xMid, s.yMid)
vtkfile["dose"] = dose
vtkfile["dose_normalized"] = dose./maximum(dose)
outfiles = vtk_save(vtkfile)

println("main finished")
