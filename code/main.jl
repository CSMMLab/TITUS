using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

nx = 151;
ny = 151;
problem = "2DHighD"
s = Settings(nx,ny,50,problem);
rhoMin = minimum(s.density);

E = [1,0.997378,0.994756,0.992134,0.989512,0.98689,0.984267,0.981645,0.979023,0.976401,0.973779,0.971157,0.968535,0.965913,0.963291,0.960669,0.958046,0.955424,0.952802,0.95018,0.947558,0.944936,0.942314,0.939692,0.93707,0.934448,0.931825,0.929203,0.926581,0.923959,0.921337,0.918715,0.916093,0.913471,0.910849,0.908227,0.905605,0.902982,0.90036,0.897718,0.895072,0.892426,0.88978,0.887134,0.884488,0.881842,0.879196,0.87655,0.873904,0.871258,0.868612,0.865966,0.86332,0.860674,0.858028,0.855382,0.852736,0.85009,0.847444,0.844798,0.842152,0.839506,0.83686,0.834214,0.831568,0.828922,0.826276,0.82363,0.820984,0.818338,0.815692,0.813046,0.8104,0.807754,0.805108,0.802462,0.799813,0.797132,0.794451,0.791769,0.789088,0.786407,0.783725,0.781044,0.778363,0.775682,0.773,0.770319,0.767638,0.764957,0.762275,0.759594,0.756913,0.754231,0.75155,0.748869,0.746188,0.743506,0.740825,0.738144,0.735463,0.732781,0.7301,0.727419,0.724737,0.722056,0.719375,0.716694,0.714012,0.711331,0.70865,0.705968,0.703287,0.700606,0.697884,0.69515,0.692416,0.689682,0.686949,0.684215,0.681481,0.678747,0.676013,0.673279,0.670546,0.667812,0.665078,0.662344,0.65961,0.656876,0.654143,0.651409,0.648675,0.645941,0.643207,0.640473,0.637739,0.635006,0.632272,0.629538,0.626804,0.62407,0.621336,0.618603,0.615869,0.613135,0.610401,0.607667,0.604933,0.6022,0.59945,0.596636,0.593822,0.591008,0.588194,0.58538,0.582566,0.579752,0.576938,0.574123,0.571309,0.568495,0.565681,0.562867,0.560053,0.557239,0.554425,0.551611,0.548797,0.545983,0.543169,0.540355,0.537541,0.534727,0.531913,0.529099,0.526284,0.52347,0.520656,0.517842,0.515028,0.512214,0.5094,0.506586,0.503772,0.500958,0.498088,0.49519,0.492292,0.489393,0.486495,0.483597,0.480698,0.4778,0.474902,0.472003,0.469105,0.466207,0.463308,0.46041,0.457512,0.454614,0.451715,0.448784,0.445806,0.442828,0.43985,0.436872,0.433894,0.430916,0.427938,0.42496,0.421982,0.419004,0.416026,0.413048,0.41007,0.407092,0.404114,0.401136,0.398093,0.39501,0.391926,0.388843,0.38576,0.382676,0.379593,0.37651,0.373426,0.370343,0.36726,0.364176,0.361093,0.35801,0.354926,0.351843,0.348702,0.345475,0.342248,0.339021,0.335794,0.332567,0.32934,0.326113,0.322886,0.319659,0.316431,0.313204,0.309977,0.30675,0.303523,0.300296,0.296885,0.293454,0.290024,0.286594,0.283163,0.279733,0.276303,0.272873,0.269442,0.266012,0.262582,0.259151,0.255721,0.252291,0.24876,0.245028,0.241295,0.237563,0.23383,0.230098,0.226366,0.222633,0.218901,0.215168,0.211436,0.207704,0.203971,0.200239,0.196203,0.192146,0.18809,0.184033,0.179976,0.17592,0.171624,0.167258,0.162893,0.158527,0.154161,0.149776,0.144989,0.140203,0.135416,0.13063,0.125843,0.120559,0.115169,0.109778,0.104388,0.0988884,0.0929106,0.0867031,0.0802774,0.0733155,0.0659502,0.0580223,0.049277,0.0391109,0.0263731,5e-05]

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
#X_dlr,S_dlr,W_dlr, dose_DLR, psi_DLR = SolveFirstCollisionSourceDLR(solver1);
u, dose_DLR = Solve(solver1);
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);
u = Vec2Mat(s.NCellsX,s.NCellsY,u[:,1]);

X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

XRef = (xRef[2:end-1]'.*ones(size(xRef[2:end-1])))
YRef = (yRef[2:end-1]'.*ones(size(yRef[2:end-1])))'

XMC = (xMC[2:end-1]'.*ones(size(xMC[2:end-1])))
YMC = (yMC[2:end-1]'.*ones(size(yMC[2:end-1])))'

fig = figure("Dose, DLRA",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
#pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]',vmin=0.0,vmax=maximum(dose_DLR[2:end-1,2:end-1]))
pcolormesh(Y,X,dose_DLR[2:end-1,2:end-1]')
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
pcolormesh(YRef,XRef,doseRef[2:end-1,2:end-1]')
ax.tick_params("both",labelsize=20) 
#colorbar()
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title(L"dose, Starmap", fontsize=25)
tight_layout()
savefig("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

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
