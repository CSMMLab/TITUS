using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles

close("all")

nx = 201;
s = Settings(nx,nx,50);
s.xMid = s.xMid./s.b
s.yMid = s.yMid./s.d
s.b = 1.0
s.d = 1.0
rhoMin = minimum(s.density);

## read density
density = Float64.(Gray.(load("liver_cut.jpg")))
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD));
x = collect(range(s.c,stop = s.d,length = nyD));
############################
s.nPN = 2;
solver = SolverCSD(s);

X_dlr = readdlm("output/liverSmallPerson/X_csd_1stcollision_DLRA_Rank50nx200ny200nPN21eMax2.0rhoMin0.1epsAdapt0.001.txt", Float64)

# all contours magma
fig, (ax1, ax3, ax2, ax4) = plt.subplots(2, 2,figsize=(15,13),dpi=100)
CS1 = ax1.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,1])[2:(end-1),2:(end-1)],cmap="plasma")
CS2 = ax2.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,2])[2:(end-1),2:(end-1)],cmap="plasma")
CS3 = ax3.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,3])[2:(end-1),2:(end-1)],cmap="plasma")
CS4 = ax4.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,4])[2:(end-1),2:(end-1)],cmap="plasma")
ax1.set_title(L"X_1", fontsize=20)
ax2.set_title(L"X_2", fontsize=20)
ax3.set_title(L"X_3", fontsize=20)
ax4.set_title(L"X_4", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax3.tick_params("both",labelsize=15) 
ax4.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax3.set_xlabel("x / [cm]", fontsize=15)
ax3.set_ylabel("y / [cm]", fontsize=15)
ax4.set_xlabel("x / [cm]", fontsize=15)
ax4.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
ax3.set_aspect(1)
ax4.set_aspect(1)
cb = plt.colorbar(CS1,fraction=0.035, pad=0.02, ax=ax1)
cb.ax.tick_params(labelsize=15)
cb = plt.colorbar(CS2,fraction=0.035, pad=0.02, ax=ax2)
cb.ax.tick_params(labelsize=15)
cb = plt.colorbar(CS3,fraction=0.035, pad=0.02, ax=ax3)
cb.ax.tick_params(labelsize=15)
cb = plt.colorbar(CS4,fraction=0.035, pad=0.02, ax=ax4)
cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/X_compare_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")

println("main finished")
