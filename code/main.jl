using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

nx = 71;
s = Settings(nx,nx,100);

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
solver1 = SolverCSD(s);
X,S,W, dose, rankInTime = SolveFirstCollisionSourceAdaptiveDLR(solver1);
#@time X,S,W, dose = SolveMCollisionSourceDLR(solver1);
#u, dose = SolveFirstCollisionSource(solver1);
dose = Vec2Mat(s.NCellsX,s.NCellsY,dose);

s = Settings(nx,nx,100);
#s = Settings(nx,nx,int(maximum(rankInTime[2,:])));
solver2 = SolverCSD(s);
X_dlr,S_dlr,W_dlr, dose_DLR = SolveFirstCollisionSourceDLR(solver2);
#X_dlr,S_dlr,W_dlr, dose_DLR = SolveMCollisionSourceDLR(solver2);
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);

s3 = Settings(nx,nx,50);
solver3 = SolverCSD(s3);
X_dlrM,S_dlrM,W_dlrM, dose_DLRM = SolveMCollisionSourceDLR(solver3);
dose_DLRM = Vec2Mat(s3.NCellsX,s3.NCellsY,dose_DLRM);

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

fig = figure("Dose, DLRA-M",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid,s.yMid,dose_DLRM,vmin=0.0,vmax=maximum(dose))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRAM", fontsize=25)
savefig("output/doseDLRAMNx$(s.Nx)")

fig = figure("Dose countours, full",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(solver1.density,cmap="gray")
contour(dose, 30,cmap="magma")
#colorbar()
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)

fig = figure("Dose countours, DLRA",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(solver2.density,cmap="gray")
contour(dose_DLR, 30,cmap="magma")
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)

fig = figure("Dose countours, DLRAM",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(solver2.density,cmap="gray")
contour(dose_DLRM, 30,cmap="magma")
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)

#colorbar()
savefig("output/densityDLRNx$(s.Nx)")

fig = figure("X_1",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,1]))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
#CS = plt.pcolormesh(X, Y, Z)

fig = figure("X_2",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,2]))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
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

fig = figure("rank in energy",figsize=(10, 10), dpi=100)
ax = gca()
ax.plot(rankInTime[1,:],rankInTime[2,:], "b--", linewidth=2, label=L"$\bar{\vartheta} = 0.05$", alpha=1.0)
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper left", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("rank_in_time_euler_nx$(s.NCellsX).png")

# write vtk file
vtkfile = vtk_grid("output/dose_csd_nx$(s.NCellsX)ny$(s.NCellsY)", s.xMid, s.yMid)
vtkfile["dose"] = dose
vtkfile["dose_normalized"] = dose./maximum(dose)
outfiles = vtk_save(vtkfile)

rhoMin = minimum(s.density);
writedlm("output/dose_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", dose)
writedlm("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", dose_DLR)
writedlm("output/dose_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", dose_DLRM)

writedlm("output/X_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", X_dlr)
writedlm("output/X_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", X_dlrM)

writedlm("output/S_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", S_dlr)
writedlm("output/S_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", S_dlrM)

writedlm("output/W_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", W_dlr)
writedlm("output/W_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", W_dlrM)

writedlm("output/u_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax(s.eMax)rhoMin$(rhoMin).txt", u)

println("main finished")
