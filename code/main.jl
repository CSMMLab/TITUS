using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

nx = 51;
ny = 51;
nz = 51;
problem = "LineSource"
s = Settings(nx,ny,nz,200,problem);
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

L1 = 1;
s = Settings(nx,ny,nz,200,problem);
solver2 = SolverMLCSD(s,L1);
X,S,W, dose, rankInTime, psi = SolveMCollisionSourceDLR(solver2);
doseTensor = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose);

dose = doseTensor[:,:,int(floor(s.NCellsZ/2))]
zMid = int(floor(s.NCellsZ/2));
density = solver2.density[2:end-1,2:end-1,zMid]'

X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'
levels = 100;

fig = figure("Dose, adaptive DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,dose[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose[2:end-1,2:end-1]))
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, $P_N$", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)$(s.epsAdapt).png")

fig = figure("Dose countours, DLRA-M=1",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(Y,X,density,cmap="gray")
contour(Y,X,dose[2:end-1,2:end-1]', levels,cmap="plasma")
#colorbar()
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
savefig("output/doseiso_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")

# line plot dose
fig, ax = subplots()
#nyRef = length(yRef)
ax.plot(s.xMid,dose[:,Int(floor(s.NCellsY/2))]./maximum(dose[:,Int(floor(s.NCellsY/2))]), "r--", linewidth=2, label="CSD", alpha=0.8)
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
ltype = ["b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
for l = 1:L1
    ax.plot(rankInTime[1,1:(end-1)],rankInTime[l+1,1:(end-1)], ltype[l], linewidth=2, label="collision $(l)", alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper left", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/rank_in_energy_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)$(s.epsAdapt).png")

# plot scalar flux
if s.problem == "LineSource"
    scalarFlux = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,Ten2Vec(psi)*solver2.M')[2:(end-1),2:(end-1),zMid,1];
    for l = 1:L1
        r = int(rankInTime[end,l])
        scalarFlux .+= Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,solver2.X[l,:,1:r]*solver2.S[l,1:r,1:r]*solver2.W[l,1,1:r])[2:(end-1),2:(end-1),zMid];
    end
    fig = figure("scalar flux, L = 2",figsize=(10,10),dpi=100)
    ax = gca()
    pcolormesh(Y,X,scalarFlux')
    ax.tick_params("both",labelsize=20) 
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    tight_layout()
    #CS = plt.pcolormesh(X, Y, Z)
    savefig("output/scalarflux1_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")    
end

# write vtk file
vtkfile = vtk_grid("output/dose_csd_nx$(s.NCellsX)ny$(s.NCellsY)$(s.NCellsZ)", s.xMid, s.yMid, s.zMid)
vtkfile["dose"] = doseTensor
vtkfile["dose_normalized"] = doseTensor./maximum(doseTensor)
outfiles = vtk_save(vtkfile)

println("main finished")
