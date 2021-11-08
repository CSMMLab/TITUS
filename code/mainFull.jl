using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

nx = 201;
s = Settings(nx,nx,100);
rhoMin = minimum(s.density);

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
solver = SolverCSD(s);
u, doseFull = SolveFirstCollisionSource(solver);
doseFull = Vec2Mat(s.NCellsX,s.NCellsY,doseFull);

L1 = 2;
s = Settings(nx,nx,200);
solver1 = SolverMLCSD(s,L1);
X,S,W, dose, rankInTime, psi = SolveMCollisionSourceDLR(solver1);
dose = Vec2Mat(s.NCellsX,s.NCellsY,dose);

s = Settings(nx,nx,50);
#s = Settings(nx,nx,int(maximum(rankInTime[2,:])));
solver2 = SolverCSD(s);
X_dlr,S_dlr,W_dlr, dose_DLR, psi = SolveFirstCollisionSourceDLR(solver2);
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);

L = 4
s3 = Settings(nx,nx,200);
solver3 = SolverMLCSD(s3,L);
X_dlrM,S_dlrM,W_dlrM, dose_DLRM, rankInTimeML, psi = SolveMCollisionSourceDLR(solver3);
dose_DLRM = Vec2Mat(s3.NCellsX,s3.NCellsY,dose_DLRM);

fig = figure("Dose, full",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],doseFull[2:end,2:end],vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, P$_N$", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_pn_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, adaptive DLRA",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],dose[2:end,2:end],vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, adaptive DLRA", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")

fig = figure("Dose, DLRA",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],dose_DLR[2:end,2:end],vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, DLRA-M",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],dose_DLRM[2:end,2:end],vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRAM", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRAM_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose countours, full",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],solver1.density[2:end,2:end],cmap="gray")
contour(s.xMid[2:end],s.yMid[2:end],doseFull[2:end,2:end], 30,cmap="magma",vmin=0.0,vmax=maximum(dose[2:end,2:end]))
#colorbar()
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseiso_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose countours, adaptive",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],solver1.density[2:end,2:end],cmap="gray")
contour(s.xMid[2:end],s.yMid[2:end],dose[2:end,2:end], 30,cmap="magma",vmin=0.0,vmax=maximum(dose[2:end,2:end]))
#colorbar()
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseiso_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")

fig = figure("Dose countours, DLRA",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],solver2.density[2:end,2:end],cmap="gray")
contour(s.xMid[2:end],s.yMid[2:end],dose_DLR[2:end,2:end], 30,cmap="magma",vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseiso_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose countours, DLRAM",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],solver2.density[2:end,2:end],cmap="gray")
contour(s.xMid[2:end],s.yMid[2:end],dose_DLRM[2:end,2:end], 30,cmap="magma",vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseiso_csd_1stcollision_DLRAM_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(23,10),dpi=100)
ax1.pcolormesh(s.xMid[2:end],s.yMid[2:end],solver1.density[2:end,2:end],cmap="gray")
CS = ax1.contour(s.xMid[2:end],s.yMid[2:end],dose_DLR[2:end,2:end], 30,cmap="magma",vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax2.pcolormesh(s.xMid[2:end],s.yMid[2:end],solver2.density[2:end,2:end],cmap="gray")
ax2.contour(s.xMid[2:end],s.yMid[2:end],dose[2:end,2:end], 30,cmap="magma",vmin=0.0,vmax=maximum(dose[2:end,2:end]))
ax1.set_title("fixed rank r = $(s.r)", fontsize=25)
ax2.set_title("adaptive rank", fontsize=25)
ax1.tick_params("both",labelsize=20) 
ax2.tick_params("both",labelsize=20) 
ax1.set_xlabel("x", fontsize=20)
ax1.set_ylabel("y", fontsize=20)
ax2.set_xlabel("x", fontsize=20)
ax2.set_ylabel("y", fontsize=20)
ax1.set_aspect(1)
ax2.set_aspect(1)
#colorbar(CS)
cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/doseiso_compare_csd_1stcollision_DLRAM_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")

# different contours
doseMax1 = maximum(dose_DLR[2:(end-1),2:(end-1)])
doseMax2 = maximum(dose[2:(end-1),2:(end-1)])
doseMax3 = maximum(dose_DLRM[2:(end-1),2:(end-1)])
doseMax4 = maximum(doseFull[2:(end-1),2:(end-1)])
#levels = [0.025,0.05, 0.1, 0.25, 0.5, 0.7, 0.8, .9, .95, .98];
levels = [0.15,0.2,0.25, 0.5, 0.7, 0.8, .9, .95, .98];
fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2,figsize=(23,23),dpi=100)
ax1.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],solver1.density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose_DLR[2:(end-1),2:(end-1)]./doseMax1,levels,cmap="plasma",vmin=minimum(levels),vmax=maximum(levels))
ax2.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],solver2.density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose[2:(end-1),2:(end-1)]./doseMax2,levels,cmap="plasma",vmin=minimum(levels),vmax=maximum(levels))
ax3.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],solver1.density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax3.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose_DLRM[2:(end-1),2:(end-1)]./doseMax1,levels,cmap="plasma",vmin=minimum(levels),vmax=maximum(levels))
ax4.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],solver1.density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax4.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],doseFull[2:(end-1),2:(end-1)]./doseMax1,levels,cmap="plasma",vmin=minimum(levels),vmax=maximum(levels))
ax1.set_title("fixed rank r = $(s.r)", fontsize=25)
ax2.set_title("adaptive rank", fontsize=25)
ax3.set_title("ML-DLRA fixed rank r = $(s.r)", fontsize=25)
ax4.set_title("full solution", fontsize=25)
ax1.tick_params("both",labelsize=20) 
ax2.tick_params("both",labelsize=20) 
ax3.tick_params("both",labelsize=20) 
ax4.tick_params("both",labelsize=20) 
ax1.set_xlabel("x", fontsize=20)
ax1.set_ylabel("y", fontsize=20)
ax2.set_xlabel("x", fontsize=20)
ax2.set_ylabel("y", fontsize=20)
ax3.set_xlabel("x", fontsize=20)
ax3.set_ylabel("y", fontsize=20)
ax4.set_xlabel("x", fontsize=20)
ax4.set_ylabel("y", fontsize=20)
ax1.set_aspect(1)
ax2.set_aspect(1)
ax3.set_aspect(1)
ax4.set_aspect(1)
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/doseiso_compare_csd_1stcollision_DLRAM_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")


fig = figure("X_1",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,1])[2:end,2:end])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/X1_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("X_2",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,2])[2:end,2:end])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/X2_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("X_3",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,3])[2:end,2:end])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/X3_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

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
tight_layout()
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
savefig("output/rank_in_energy_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)$(s.epsAdapt).png")

fig = figure("rank in energy, ML",figsize=(10, 10), dpi=100)
ax = gca()
ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[2,1:(end-1)], "b-", linewidth=2, label="1st collision", alpha=1.0)
ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[3,1:(end-1)], "r-", linewidth=2, label="2nd collision", alpha=1.0)
ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[4,1:(end-1)], "m-", linewidth=2, label="3rd collision", alpha=1.0)
ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[5,1:(end-1)], "k-", linewidth=2, label="collided", alpha=1.0)
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper left", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/rank_in_energy_ML_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")

writedlm("output/dose_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).txt", dose)
writedlm("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_DLR)
writedlm("output/dose_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_DLRM)

writedlm("output/X_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).txt", X_dlr)
writedlm("output/X_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", X_dlrM)

writedlm("output/S_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", S_dlr)
writedlm("output/S_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", S_dlrM)

writedlm("output/W_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", W_dlr)
writedlm("output/W_csd_1stcollision_DLRAM_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", W_dlrM)

writedlm("output/rank_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", rankInTime)

writedlm("output/u_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", u)
writedlm("output/dose_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", doseFull)

println("main finished")
