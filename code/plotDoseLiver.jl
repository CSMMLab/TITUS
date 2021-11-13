using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles

close("all")

nx = 201;
s = Settings(nx,nx,50,"liver");
s.xMid = s.xMid
s.yMid = s.yMid

rhoMin = minimum(s.density);
## read density
density = Float64.(Gray.(load("liver_cut.jpg")))
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD));
x = collect(range(s.c,stop = s.d,length = nyD));
############################
s.nPN = 5;
solver = SolverCSD(s);

writedlm("output/muNPN$(s.nPN).txt", solver.Q.pointsmuphi[:,1])
writedlm("output/phiNPN$(s.nPN).txt", solver.Q.pointsmuphi[:,2])

#dose = readdlm("output/lung/dose_csd_1stcollision_nx200ny200nPN21eMax21.0rhoMin0.050.005.txt", Float64)
dose_dlra = readdlm("outputLiver/dose_csd_1stcollision_DLRA_problemliver_Rank50_nx200ny200nPN21eMax60.0rhoMin0.05.txt", Float64)
dose_Llow = readdlm("outputLiver/dose_csd_1stcollision_DLRA_problemliver_nx200ny200nPN21eMax60.0rhoMin0.05L2epsAdapt0.05.txt", Float64)
dose_Lhigh = readdlm("outputLiver/dose_csd_1stcollision_DLRA_problemliver_nx200ny200nPN21eMax60.0rhoMin0.05L10epsAdapt0.05.txt", Float64)
dose_full = readdlm("outputLiver/dose_csd_1stcollision_problemliver_nx200ny200nPN21eMax60.0rhoMin0.05.txt", Float64)

# all contours magma
doseMax1 = maximum(dose_dlra[2:(end-1),2:(end-1)])
doseMax2 = maximum(dose_Llow[2:(end-1),2:(end-1)])
doseMax3 = maximum(dose_Lhigh[2:(end-1),2:(end-1)])
doseMax4 = maximum(dose_full[2:(end-1),2:(end-1)])
levels = 40;
fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2,figsize=(15,15),dpi=100)
ax1.pcolormesh(x[2:(end-1)],y[2:(end-1)],density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose_dlra[2:(end-1),2:(end-1)]./doseMax1,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(x[2:(end-1)],y[2:(end-1)],density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose_Llow[2:(end-1),2:(end-1)]./doseMax2,levels,cmap="plasma",vmin=0,vmax=1)
ax3.pcolormesh(x[2:(end-1)],y[2:(end-1)],density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax3.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose_Lhigh[2:(end-1),2:(end-1)]./doseMax3,levels,cmap="plasma",vmin=0,vmax=1)
ax4.pcolormesh(x[2:(end-1)],y[2:(end-1)],density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax4.contour(s.xMid[2:(end-1)],s.yMid[2:(end-1)],dose_full[2:(end-1),2:(end-1)]./doseMax4,levels,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("fixed rank r = $(s.r)", fontsize=20)
ax2.set_title("adaptive rank", fontsize=20)
ax3.set_title("ML-DLRA fixed rank r = $(s.r)", fontsize=20)
ax4.set_title("full solution", fontsize=20)
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
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/doseiso_compare_csd_1stcollision_DLRAM_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)epsAdapt$(s.epsAdapt).png")

println("main finished")
