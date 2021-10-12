using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

s = Settings(151,151,20);

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
elseif s.problem == "2DHighD"
    doseRef = readdlm("validationData/dose_starmap_full301_inhomogenity.txt", Float64)
    xRef = readdlm("validationData/x_starmap_nx301.txt", Float64)
    yRef = readdlm("validationData/y_starmap_ny301.txt", Float64)
else
    xRef = 0; doseRef = 1;
end

############################
solver1 = SolverCSD(s)
solver2 = SolverCSD(s)

@time u, dose = SolveFirstCollisionSource(solver1);

@time u_DLR, dose_DLR = SolveFirstCollisionSourceDLR(solver2);
#@time u, dose = Solve(solver);

u = Vec2Mat(s.NCellsX,s.NCellsY,u)
dose = Vec2Mat(s.NCellsX,s.NCellsY,dose)

u_DLR = Vec2Mat(s.NCellsX,s.NCellsY,u_DLR)
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR)

fig = figure("Dose Difference",figsize=(10,10),dpi=100)

pcolormesh(dose-dose_DLR)
#colorbar()
savefig("output/doseDiffNx$(s.Nx)")

fig = figure("Dose, full",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid,s.yMid,dose,vmin=0.0,vmax=maximum(dose))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, $P_N$", fontsize=25)
savefig("output/doseNx$(s.Nx)")

fig = figure("Dose, DLRA",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid,s.yMid,dose_DLR,vmin=0.0,vmax=maximum(dose))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
savefig("output/doseDLRANx$(s.Nx)")

fig = figure("Dose countours, full",figsize=(10,10),dpi=100)

pcolormesh(solver1.density,cmap="gray")
contour(dose, 30,cmap="magma")
#colorbar()

fig = figure("Dose countours, DLRA",figsize=(10,10),dpi=100)

pcolormesh(solver2.density,cmap="gray")
contour(dose_DLR, 30,cmap="magma")
#colorbar()
savefig("output/densityDLRNx$(s.Nx)")

fig = figure("density",figsize=(10,10),dpi=100)


fig = figure("u Contour",figsize=(10,10),dpi=100)

pcolormesh(u[:,:,1])
#CS = plt.pcolormesh(X, Y, Z)

# line plot dose
fig, ax = subplots()
#nyRef = length(yRef)
ax.plot(s.xMid,dose[:,Int(floor(s.NCellsY/2))]./maximum(dose[:,Int(floor(s.NCellsY/2))]), "r--", linewidth=2, label="CSD", alpha=0.8)
ax.plot(s.xMid,dose_DLR[:,Int(floor(s.NCellsY/2))]./maximum(dose_DLR[:,Int(floor(s.NCellsY/2))]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
if s.problem == "2DHighD"
 #   ax.plot(xRef',doseRef[:,Int(floor(nyRef/2))]./maximum(doseRef[:,Int(floor(nyRef/2))]), "k-", linewidth=2, label="Starmap", alpha=0.6)
end
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

writedlm("output/dose_csd_nx$(s.NCellsX)ny$(s.NCellsY).txt", dose)

println("main finished")
