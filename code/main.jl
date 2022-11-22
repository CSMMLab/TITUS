using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using Trapz

close("all")

info = "CUDA"

nx = Int(floor(2 * 50));
ny = Int(floor(8 * 50));
nz = Int(floor(2 * 50));
problem ="validation" #"2DHighD"
particle = "Protons"
s = Settings(nx,ny,nz,5,problem, particle);
rhoMin = minimum(s.density);

if s.problem == "validation"
    nx_MC = 300;
    doseRef = zeros(nx_MC,nx_MC,nx_MC);
    read!("validationData/proton_validation3D_dose.bin",doseRef)
    xRef = collect(range(0,2,nx_MC));
    yRef = collect(range(0,2,nx_MC));
    zRef = collect(range(0,8,nx_MC));
    idxXref = 150;
    idxYref = 150;
    idxZref = 150;
else
    xRef = 0; doseRef = 1;
end
############################

solver1 = SolverCSD(s);
X_dlr,S_dlr,W_dlr_SN,W_dlr, dose_DLR, psi_DLR = CudaFullSolveFirstCollisionSourceDLR4thOrder(solver1);
#X_dlr,S_dlr,W_dlr_SN,W_dlr, dose_DLR, psi_DLR = SolveFirstCollisionSourceDLR4thOrderFP(solver1);
#u, dose_DLR,psi = SolveFirstCollisionSource(solver1);
u = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,X_dlr*Diagonal(S_dlr)*W_dlr[1,:]);
dose_DLR = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_DLR);
R0 = (0.022 * (s.eMax - s.eRest)^1.77)/10 #Bragg-Kleemann rule with parameters acc. to Ulmer et al. 2011
idx_0 = ceil(Int,s.NCellsY/s.d*(0 + s.y0))
sigma_z = 0.012.* R0^(0.935) #std of range straggling convolution kernel (Bortfeld 1997)
z = collect(1:s.NCellsY).*s.d ./s.NCellsY .- s.y0
dose_DLRconv = zeros(size(dose_DLR))
for j=1:s.NCellsX
     for k=1:s.NCellsZ
         for i=1:s.NCellsY
            dose_DLRconv[j,i,k]=trapz(z[idx_0:end], normpdf.(z[idx_0:end],z[i],sigma_z).* dropdims(dose_DLR[j,idx_0:end,k], dims = tuple(findall(size(dose_DLR[j,idx_0:end,k]) .== 1)...))); 
         end
     end
end
dose_DLR = dose_DLRconv
idxX = Int(floor(s.NCellsX/2))
idxY = Int(floor(s.NCellsY/2))   #idxY = floor(Int,s.NCellsY/s.d*(0.5*s.d + s.y0))
idxZ = Int(floor(s.NCellsZ/2))

X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'
Z = (s.zMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
YZ = (s.yMid[2:end-1]'.*ones(size(s.zMid[2:end-1])))'


XRef = (xRef[2:end-1]'.*ones(size(xRef[2:end-1])))
YRef = (yRef[2:end-1]'.*ones(size(yRef[2:end-1])))'
ZRef = (zRef[2:end-1]'.*ones(size(zRef[2:end-1])))'

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

doseSumRef = zeros(nx_MC,nx_MC);
for k = 1:nx_MC
    doseSumRef .+= doseRef[:,k,:]
end

fig = figure("sumDose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,doseSum[2:end-1,2:end-1]',vmax=maximum(doseSum[2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\sum$ dose, DLRA", fontsize=25)
ax.set_aspect(1)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("sumDose, ref",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(ZRef,XRef,doseSumRef[2:end-1,2:end-1]',vmax=maximum(doseSumRef[2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"$\sum$ dose, ref", fontsize=25)
ax.set_aspect(1)
tight_layout()
savefig("output/doseYX_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1,idxZ]',vmax=maximum(dose_DLR[2:end-1,2:end-1,idxZ]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, DLRA", fontsize=25)
ax.set_aspect(1)
tight_layout()
savefig("output/doseYX_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

fig = figure("Dose, ref",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(ZRef,XRef,doseRef[2:end-1,idxYref,2:end-1]',vmax=maximum(doseRef[2:end-1,idxYref,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, ref", fontsize=25)
ax.set_aspect(1)
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
ax.set_aspect(1)
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
ax.set_aspect(1)
tight_layout()
savefig("output/doseYXcontour_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

# line plot dose
fig, ax = subplots()
nyRef = length(yRef)
ax.plot(s.xMid,dose_DLR[:,Int(floor(s.NCellsY/2)),idxZ]./maximum(dose_DLR[:,Int(floor(s.NCellsY/2)),idxZ]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
ax.plot(xRef,doseRef[:,idxYref,idxZref]./maximum(doseRef[:,idxYref,idxZref]), "r-", linewidth=2, label="reference", alpha=0.8)
ax.legend(loc="upper left")
#ax.set_xlim([s.xMid[1],s.xMid[end]])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/DoseCutYNx$(s.Nx)nPN$(s.nPN)$(info)")

fig, ax = subplots()
nyRef = length(yRef)
ax.plot(s.yMid,dose_DLR[Int(floor(s.NCellsX/2)),:,idxZ]./maximum(dose_DLR[Int(floor(s.NCellsX/2)),:,idxZ]), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
ax.plot(zRef .+ s.y0 ,doseRef[idxXref,idxYref,:]./maximum(doseRef[idxXref,idxYref,:]), "r-", linewidth=2, label="reference", alpha=0.8)
ax.legend(loc="upper left")
ax.set_xlim([s.yMid[1],s.yMid[end]])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/DoseCutXNx$(s.Nx)nPN$(s.nPN)$(info)")

doseIntegrated = zeros(ny-1);
for j = 1:ny-1
    doseIntegrated[j] = sum( dose_DLR[idxX,j,:] )
end
doseIntegratedRef = zeros(nx_MC);
for j = 1:nx_MC
    doseIntegratedRef[j] = sum( doseRef[idxXref,:,j] )
end

fig, ax = subplots()
nyRef = length(yRef)
ax.plot(s.yMid,doseIntegrated./maximum(doseIntegrated), "b--", linewidth=2, label="CSD_DLR", alpha=0.8)
ax.plot(zRef .+ s.y0,doseIntegratedRef./maximum(doseIntegratedRef), "r-", linewidth=2, label="reference", alpha=0.8)
ax.legend(loc="upper left")
ax.set_xlim([s.yMid[1],s.yMid[end]])
ax.set_ylim([0,1.05])
ax.tick_params("both",labelsize=20) 
show()
tight_layout()
savefig("output/IntegratedDoseCutXNx$(s.Nx)nPN$(s.nPN)$(info)")

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
ax.set_aspect(1)
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
ax.set_aspect(1)
tight_layout()
savefig("output/uYX_csd_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nz$(s.NCellsZ)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

println("main finished")
