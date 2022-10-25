using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

info = "GPU"

nx = Int(floor(2 * 50));
ny = Int(floor(8 * 50));
nz = Int(floor(2 * 50));
problem ="validation" #"2DHighD"
particle = "Protons"
s = Settings(nx,ny,nz,5,problem, particle);
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
elseif s.problem == "validation"
    u_MC = zeros(200,200,200);
    read!("validationData/proton_validation_doseMC.bin",u_MC)
    xRef = 0; doseRef = 1;
end

############################

solver1 = SolverCSD(s);
X_dlr,S_dlr,W_dlr_SN,W_dlr, dose_DLR, psi_DLR = CudaFullSolveFirstCollisionSourceDLR4thOrder(solver1);
#X_dlr,S_dlr,W_dlr_SN,W_dlr, dose_DLR, psi_DLR = CudaSolveFirstCollisionSourceDLR4thOrderSN(solver1);
#u, dose_DLR,psi = SolveFirstCollisionSource(solver1);
u = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,X_dlr*Diagonal(S_dlr)*W_dlr[1,:]);
dose_DLR = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_DLR);
idxX = Int(floor(s.NCellsX/2))
idxY = Int(floor(s.NCellsY/2))
idxZ = Int(floor(s.NCellsZ/2))

X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'
Z = (s.zMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
YZ = (s.yMid[2:end-1]'.*ones(size(s.zMid[2:end-1])))'


XRef = (xRef[2:end-1]'.*ones(size(xRef[2:end-1])))
YRef = (yRef[2:end-1]'.*ones(size(yRef[2:end-1])))'

# write vtk file
vtkfile = vtk_grid("output/dose_csd_nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)", s.xMid, s.yMid,s.zMid)
vtkfile["dose"] = dose_DLR
#vtkfile["dose_normalized"] = dose_DLR./maximum(dose_DLR)
#vtkfile["u"] = u/0.5/sqrt(solver1.gamma[1])
outfiles = vtk_save(vtkfile)

doseSum = zeros(nx-1,ny-1);
for k = 1:nz-1
    doseSum .+= dose_DLR[:,:,k]
end

fig = figure("sumDose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,doseSum[2:end-1,2:end-1]',vmax=maximum(doseSum[2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\sum$ dose, DLRA", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")


fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1,idxZ]',vmax=maximum(dose_DLR[2:end-1,2:end-1,idxZ]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
tight_layout()
savefig("output/doseYX_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, DLRA cut z",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(YZ',Z',dose_DLR[idxX,2:end-1,2:end-1]',vmax=maximum(dose_DLR[idxX,2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
tight_layout()
savefig("output/doseYZ_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

levels = 20;
fig = figure("Dose countours, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,solver1.density[2:end-1,2:end-1,idxZ]',cmap="gray")
contour(Y,X,dose_DLR[2:end-1,2:end-1,idxZ]', levels,cmap="plasma")
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseYXcontour_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

# line plot dose
fig, ax = subplots()
nyRef = length(yRef)
ax.plot(s.xMid,dose_DLR[:,Int(floor(s.NCellsY/2)),idxZ]./maximum(dose_DLR[:,Int(floor(s.NCellsY/2)),idxZ]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
ax.legend(loc="upper left")
ax.set_xlim([s.xMid[1],s.xMid[end]])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/DoseCutYNx$(s.Nx)nPN$(s.nPN)$(info)")

fig, ax = subplots()
nyRef = length(yRef)
ax.plot(s.yMid,dose_DLR[Int(floor(s.NCellsX/2)),:,idxZ]./maximum(dose_DLR[Int(floor(s.NCellsX/2)),:,idxZ]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
ax.legend(loc="upper left")
ax.set_xlim([s.yMid[1],s.yMid[end]])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/DoseCutXNx$(s.Nx)nPN$(s.nPN)$(info)")

uSum = zeros(nx-1,ny-1);
for k = 1:nz-1
    uSum .+= u[:,:,k]
end

fig = figure("sum u",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(Y,X,uSum[2:end-1,2:end-1]')
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\sum$ u", fontsize=25)
tight_layout()
savefig("output/sum_uYX_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")


fig = figure("u",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(Y,X,u[2:end-1,2:end-1,idxZ]')
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"u", fontsize=25)
tight_layout()
savefig("output/uYX_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

println("main finished")
