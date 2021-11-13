using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

problem = "2DHighD"
nx = 501;
s = Settings(nx,nx,100,problem);
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
    nxRef = length(xRef)
    nyRef = length(yRef)
else
    xRef = 0; doseRef = 1;
end

dose_dlra = readdlm("output/dose_csd_1stcollision_DLRA_problem2DHighD_Rank50_nx200ny200nPN21eMax1.0rhoMin1.0.txt", Float64)

s = Settings(nx,nx,50,problem);
s.nPN = 13;
#s = Settings(nx,nx,int(maximum(rankInTime[2,:])));
solver2 = SolverCSD(s);
X_dlr,S_dlr,W_dlr, dose_DLR = SolveUnconventional(solver2);
dose_dlra = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);

s = Settings(nx,nx,50,problem);
s.nPN = 13;
#s = Settings(nx,nx,int(maximum(rankInTime[2,:])));
solver2 = SolverCSD(s);
X_fc,S_fc,W_fc, dose_fc = SolveFirstCollisionSourceDLR(solver2);
dose_fc = Vec2Mat(s.NCellsX,s.NCellsY,dose_fc);

fig = figure("Dose, DLRA",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],dose_dlra[2:end,2:end],vmin=0.0,vmax=maximum(dose_dlra[2:end,2:end]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

# line plot dose
fig, ax = subplots()
#nyRef = length(yRef)
#ax.plot(s.xMid,dose[:,Int(floor(s.NCellsY/2))]./maximum(dose[:,Int(floor(s.NCellsY/2))]), "r--", linewidth=2, label="CSD", alpha=0.8)
ax.plot(s.xMid,dose_dlra[:,Int(floor(s.NCellsY/2))]./maximum(dose_dlra[:,Int(floor(s.NCellsY/2))]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
ax.plot(s.xMid,dose_fc[:,Int(floor(s.NCellsY/2))]./maximum(dose_fc[:,Int(floor(s.NCellsY/2))]), "r-.", linewidth=2, label="CSD_DLR, collision source", alpha=0.8)
if s.problem == "2DHighD"
   ax.plot(xRef',doseRef[:,Int(floor(nyRef/2))]./maximum(doseRef[:,Int(floor(nyRef/2))]), "k-", linewidth=2, label="Starmap", alpha=0.6)
end
#ax.plot(csd.eGrid,csd.S, "r--o", linewidth=2, label="S", alpha=0.6)
ax.legend(loc="upper left")
ax.set_xlim([s.c,s.d])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/DoseCutYNx$(s.Nx)")