using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

close("all")

problem = "LineSource"
nx = 201;
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
else
    xRef = 0; doseRef = 1;
end

############################
solver = SolverCSD(s);
u, doseFull, psi_full = SolveFirstCollisionSource(solver);
doseFull = Vec2Mat(s.NCellsX,s.NCellsY,doseFull);

s = Settings(nx,nx,50,problem);
#s = Settings(nx,nx,int(maximum(rankInTime[2,:])));
solver2 = SolverCSD(s);
X_dlr,S_dlr,W_dlr, dose_DLR, psi_DLR = SolveFirstCollisionSourceDLR(solver2);
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);

L1 = 2;
s2 = Settings(nx,nx,400,problem);
solver1 = SolverMLCSD(s2,L1);
X,S,W, dose, rankInTime, psi = SolveMCollisionSourceDLR(solver1);
dose = Vec2Mat(s2.NCellsX,s2.NCellsY,dose);

L = 4;
s3 = Settings(nx,nx,400,problem);
solver3 = SolverMLCSD(s3,L);
X_dlrM,S_dlrM,W_dlrM, dose_DLRM, rankInTimeML, psiML = SolveMCollisionSourceDLR(solver3);
dose_DLRM = Vec2Mat(s3.NCellsX,s3.NCellsY,dose_DLRM);



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

fig = figure("rank in energy, ML",figsize=(10, 10), dpi=100)
ax = gca()
ltype = ["b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
for l = 1:L
    ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[l+1,1:(end-1)], ltype[l], linewidth=2, label="collision $(l)", alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper left", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/rank_in_energy_ML_csd_1stcollision_adapt_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)$(s.epsAdapt).png")

# plot scalar flux
scalarFlux = Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psiML)*solver1.M')[2:(end-1),2:(end-1),1];
for l = 1:L1
    r = int(rankInTime[end,l])
    scalarFlux .+= Vec2Mat(s.NCellsX,s.NCellsY,solver1.X[l,:,1:r]*solver1.S[l,1:r,1:r]*solver1.W[l,1,1:r])[2:(end-1),2:(end-1)];
end
fig = figure("scalar flux, L = 2",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],scalarFlux)
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/scalarflux1_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")    

scalarFlux = Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psiML)*solver3.M')[2:(end-1),2:(end-1),1];
for l = 1:L
    r = int(rankInTimeML[end,l])
    scalarFlux .+= Vec2Mat(s.NCellsX,s.NCellsY,solver3.X[l,:,1:r]*solver3.S[l,1:r,1:r]*solver3.W[l,1,1:r])[2:(end-1),2:(end-1)];
end
fig = figure("scalar flux, L = 10",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],scalarFlux)
#pcolormesh(s.xMid[2:end],s.yMid[2:end],Vec2Mat(s.NCellsX,s.NCellsY,X_dlrM*diagm(S_dlrM)*(solver1.M*W_dlrM)[1,:])[2:end,2:end])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/scalarflux2_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")           

fig = figure("scalar flux",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psi_DLR)*solver1.M')[2:(end-1),2:(end-1),1]+Vec2Mat(s.NCellsX,s.NCellsY,X_dlr*diagm(S_dlr)*(solver1.M*W_dlr)[1,:])[2:(end-1),2:(end-1),1])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/scalarflux2_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png")   

fig = figure("scalar flux, full",figsize=(10,10),dpi=100)
ax = gca()
pcolormesh(s.xMid[2:end],s.yMid[2:end],Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psi_full)*solver.M')[2:(end-1),2:(end-1),1]+Vec2Mat(s.NCellsX,s.NCellsY,u[:,1])[2:(end-1),2:(end-1),1])
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
tight_layout()
#CS = plt.pcolormesh(X, Y, Z)
savefig("output/scalarflux2_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).png") 

writedlm("output/dose_csd_1stcollision_DLRA_problem$(s.problem)_Rank$(s.r)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_DLR)
writedlm("output/dose_csd_1stcollision_DLRA_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L1)epsAdapt$(s.epsAdapt).txt", dose)
writedlm("output/dose_csd_1stcollision_DLRA_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L)epsAdapt$(s.epsAdapt).txt", dose_DLRM)
writedlm("output/dose_csd_1stcollision_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", doseFull)

writedlm("output/X_csd_1stcollision_DLRA_problem$(s.problem)_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", X_dlr)
writedlm("output/X_csd_1stcollision_DLRA_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L)epsAdapt$(s.epsAdapt).txt", X_dlrM)

writedlm("output/S_csd_1stcollision_DLRA_problem$(s.problem)_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", S_dlr)
writedlm("output/S_csd_1stcollision_DLRA_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L)epsAdapt$(s.epsAdapt).txt", S_dlrM)

writedlm("output/W_csd_1stcollision_DLRA_problem$(s.problem)_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", W_dlr)
writedlm("output/W_csd_1stcollision_DLRA_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L)epsAdapt$(s.epsAdapt).txt", W_dlrM)

writedlm("output/rank_csd_1stcollision_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L1)epsAdapt$(s.epsAdapt).txt", rankInTime)
writedlm("output/rank_csd_1stcollision_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)L$(L)epsAdapt$(s.epsAdapt).txt", rankInTimeML)

writedlm("output/u_csd_1stcollision_problem$(s.problem)_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", u)

println("main finished")
