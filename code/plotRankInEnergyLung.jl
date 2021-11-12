using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles

close("all")

nx = 201;
s = Settings(nx,nx,50,"lung");
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
s.nPN = 2;
solver = SolverCSD(s);

rankInTime = readdlm("outputLung/rank_csd_1stcollision_problemlung_nx200ny200nPN21eMax21.0rhoMin0.05L2epsAdapt0.01.txt", Float64)
rankInTimeML = readdlm("outputLung/rank_csd_1stcollision_problemlung_nx200ny200nPN21eMax21.0rhoMin0.05L2epsAdapt0.001.txt", Float64)
L1 = 2;
L = 2;

fig = figure("rank in energy",figsize=(10, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = [L"rank $\mathbf{u}_{1}$",L"rank $\mathbf{u}_{c}$"]
for l = 1:L1
    ax.plot(rankInTime[1,1:(end-1)],rankInTime[l+1,1:(end-1)], ltype[l], linewidth=2, label=labelvec[l], alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper right", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/rank_in_energy_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)$(s.epsAdapt).png")

fig = figure("rank in energy, ML",figsize=(10, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = [L"rank $\mathbf{u}_{1}$",L"rank $\mathbf{u}_{c}$"]
for l = 1:L
    ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[l+1,1:(end-1)], ltype[l], linewidth=2, label=labelvec[l], alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper right", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/rank_in_energy_ML_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)$(s.epsAdapt).png")
