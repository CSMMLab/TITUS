using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

#close("all")

nx = 301;
ny = 301;
problem = "validationMC"
s = Settings(nx,ny,500,problem);
rhoMin = minimum(s.density);

if s.problem == "AirCavity"
    smapIn = readdlm("dose_ac.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
elseif s.problem == "WaterPhantomKerstin"
    smapIn = readdlm("doseStarmapWaterPhantom.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
elseif s.problem == "validation"
    doseRef = readdlm("validationData/dose_starmap_full150.txt", Float64)
    xRef = readdlm("validationData/x_starmap_nx150.txt", Float64)
    yRef = readdlm("validationData/y_starmap_ny150.txt", Float64)
elseif s.problem == "2DHighD"
    doseRef = readdlm("validationData/dose_starmap_full301_inhomogenity.txt", Float64)
    xRef = readdlm("validationData/x_starmap_nx301.txt", Float64)
    yRef = readdlm("validationData/y_starmap_ny301.txt", Float64)

    doseMC = readdlm("validationData/Dose_MC_inhomogenity.txt",',', Float64)
    nxMC = size(doseMC,1);
    nyMC = size(doseMC,1);
    xMC = collect(range( s.a,stop=s.b,length = nxMC));
    yMC = collect(range( s.c,stop=s.d,length = nxMC));
    XMC = (xMC[2:end-1]'.*ones(size(xMC[2:end-1])))
    YMC = (yMC[2:end-1]'.*ones(size(yMC[2:end-1])))'

    doseKiTRTold = readdlm("validationData/horizontal_full_simulation.csv",',', Float64)
    doseKiTRTold = doseKiTRTold[:,2];
    doseKiTRT = readdlm("validationData/horizontale_kitRT_Az.csv",',', Float64)
    doseKiTRT = doseKiTRT[:,2];
    xKiTRT = collect(range( s.a,stop=s.b,length = length(doseKiTRT)));
elseif s.problem == "2DHighLowD"
    doseRef = readdlm("validationData/dose_starmap_moreDensities301.txt", Float64)
    xRef = readdlm("validationData/x_starmap_nx301_moreDensities.txt", Float64)
    yRef = readdlm("validationData/y_starmap_ny301_moreDensities.txt", Float64)

    doseMC = readdlm("validationData/Dose_MC_inhomogenity.txt",',', Float64)
    nxMC = size(doseMC,1);
    nyMC = size(doseMC,1);
    xMC = collect(range( s.a,stop=s.b,length = nxMC));
    yMC = collect(range( s.c,stop=s.d,length = nxMC));
else
    xRef = 0; doseRef = 1;
end

############################

solver1 = SolverCSD(s);
#X_dlr,S_dlr,W_dlr, dose_DLR = Solver(solver1);
u, dose_DLR = Solve(solver1);
#u = X_dlr*diagm(S_dlr)*W_dlr';
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);
u = Vec2Mat(s.NCellsX,s.NCellsY,u[:,1]);

X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

XRef = (xRef[2:end-1]'.*ones(size(xRef[2:end-1])))
YRef = (yRef[2:end-1]'.*ones(size(yRef[2:end-1])))'

fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmax=maximum(dose_DLR[2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, ref",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(YRef,XRef,doseRef[2:end-1,2:end-1]',vmax=maximum(dose_DLR[2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, Starmap", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, ref new scale",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(YRef,XRef,doseRef[2:end-1,2:end-1]')
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, Starmap scale", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

if s.problem == "2DHighD"
    fig = figure("Dose, MC",figsize=(10*(s.d/s.b),10),dpi=100)
    ax = gca()
    #pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
    pcolormesh(YMC,XMC,doseMC[2:end-1,2:end-1])
    ax.tick_params("both",labelsize=20) 
    #colorbar()
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.title(L"dose, MC", fontsize=25)
    tight_layout()
    savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")
end
levels = 20;
fig = figure("Dose countours, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,solver1.density[2:end-1,2:end-1]',cmap="gray")
contour(Y,X,dose_DLR[2:end-1,2:end-1]', levels,cmap="plasma")
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseiso_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

# line plot dose
fig, ax = subplots()
nyRef = length(yRef)
#ax.plot(xRef,doseRef[:,Int(floor(s.NCellsY/2))]./maximum(doseRef[:,Int(floor(s.NCellsY/2))]), "r--", linewidth=2, label="CSD", alpha=0.8)
ax.plot(s.xMid,dose_DLR[:,Int(floor(s.NCellsY/2))]./maximum(dose_DLR[:,Int(floor(s.NCellsY/2))]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
if s.problem == "2DHighD"
   ax.plot(xRef',doseRef[:,Int(floor(nyRef/2))]./maximum(doseRef[:,Int(floor(nyRef/2))]), "k-", linewidth=2, label="Starmap", alpha=0.6)
   ax.plot(yMC,doseMC[Int(floor(nxMC/2)),:]./maximum(doseMC[Int(floor(nxMC/2)),:])*1.3, "r:", linewidth=2, label="MC", alpha=0.6)
   #ax.plot(xKiTRT,doseKiTRT, "g-.", linewidth=2, label="KiT-RT", alpha=0.6)
   #ax.plot(xKiTRT,doseKiTRTold, "r-.", linewidth=2, label="KiT-RT old", alpha=0.6)
end
#ax.plot(csd.eGrid,csd.S, "r--o", linewidth=2, label="S", alpha=0.6)
ax.legend(loc="upper left")
ax.set_xlim([s.c,s.d])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/DoseCutYNx$(s.Nx)")

fig = figure("u, full",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(Y,X,u[2:end-1,2:end-1]')
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"u, Full", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")


# write vtk file
vtkfile = vtk_grid("output/dose_csd_nx$(s.NCellsX)ny$(s.NCellsY)", s.xMid, s.yMid)
vtkfile["dose"] = dose_DLR
vtkfile["dose_normalized"] = dose_DLR./maximum(dose_DLR)
vtkfile["u"] = u/0.5/sqrt(solver1.gamma[1])
outfiles = vtk_save(vtkfile)

println("main finished")

include("validationData.jl")

println("distance to kit-RT: eGrid ",norm(E-solver1.csd.eGrid))
println("distance to kit-RT: eTrafo ",norm(eTrafo-solver1.csd.eTrafo))
println("distance to kit-RT: S ",norm(S-solver1.csd.S))
println("distance to kit-RT: SMid ",norm(Smid-solver1.csd.SMid))

AxPlus = readdlm("validationData/AxPlus.csv",',', Float64)
AxMinus = readdlm("validationData/AxMinus.csv",',', Float64)
println("distance to kit-RT: AxPlus ",norm(solver1.AxPlus-AxPlus))
println("distance to kit-RT: AxMinus ",norm(solver1.AxMinus-AxMinus))