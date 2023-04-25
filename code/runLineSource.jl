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

############################
solver = SolverCSD(s);
u, doseFull, psi_full = SolveFirstCollisionSource(solver);
doseFull = Vec2Mat(s.NCellsX,s.NCellsY,doseFull);

s = Settings(nx,nx,100,problem);
solver2 = SolverCSD(s);
X_dlr,S_dlr,W_dlr, dose_DLR, psi_DLR = SolveFirstCollisionSourceDLR(solver2);
dose_DLR = Vec2Mat(s.NCellsX,s.NCellsY,dose_DLR);

L1 = 2;
s2 = Settings(nx,nx,400,problem);
s2.epsAdapt = 0.01
solver1 = SolverMLCSD(s2,L1);
X,S,W, dose, rankInTime, psi = SolveMCollisionSourceDLR(solver1);
dose = Vec2Mat(s2.NCellsX,s2.NCellsY,dose);

L = 5;
s3 = Settings(nx,nx,400,problem);
s3.epsAdapt = 0.001
solver3 = SolverMLCSD(s3,L);
X_dlrM,S_dlrM,W_dlrM, dose_DLRM, rankInTimeML, psiML = SolveMCollisionSourceDLR(solver3);
dose_DLRM = Vec2Mat(s3.NCellsX,s3.NCellsY,dose_DLRM);

######################## plot scalar flux ########################

phi_Llow = Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psi)*solver1.M')[:,:,1];
for l = 1:L1
    r = int(rankInTime[end,l])
    phi_Llow .+= Vec2Mat(s.NCellsX,s.NCellsY,solver1.X[l,:,1:r]*solver1.S[l,1:r,1:r]*solver1.W[l,1,1:r]);
end

phi_Lhigh = Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psiML)*solver3.M')[:,:,1];
for l = 1:L
    r = int(rankInTimeML[end,l])
    phi_Lhigh .+= Vec2Mat(s.NCellsX,s.NCellsY,solver3.X[l,:,1:r]*solver3.S[l,1:r,1:r]*solver3.W[l,1,1:r]);
end

phi_dlra = Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psi_DLR)*solver1.M')[:,:,1]+Vec2Mat(s.NCellsX,s.NCellsY,X_dlr*diagm(S_dlr)*(solver1.M*W_dlr)[1,:])[:,:,1]
phi_full = Vec2Mat(s.NCellsX,s.NCellsY,Mat2Vec(psi_full)*solver.M')[:,:,1]+Vec2Mat(s.NCellsX,s.NCellsY,u[:,1])[:,:,1];

lsRefFull = readdlm("refPhiFull.txt", ',', Float64);
nx = size(lsRefFull,1)
xMid = collect(range(-1.5,stop = 1.5,length = nx));
yMid = collect(range(-1.5,stop = 1.5,length = nx));

fig, (ax2, ax1, ax3, ax4) = plt.subplots(2, 2,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],phi_dlra[2:(end-1),2:(end-1)],cmap="plasma")
CS = ax2.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],phi_Llow[2:(end-1),2:(end-1)],cmap="plasma")
#CS = ax3.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],phi_Lhigh[2:(end-1),2:(end-1)],cmap="plasma")
CS = ax3.pcolormesh(s.xMid[2:(end-1)],s.yMid[2:(end-1)],phi_full[2:(end-1),2:(end-1)],cmap="plasma")
CS = ax4.pcolormesh(xMid[2:(end-1)],yMid[2:(end-1)],lsRefFull[2:(end-1),2:(end-1)],cmap="plasma")
ax1.set_title(L"P$_N$", fontsize=20)
ax2.set_title(L"L = 1, $\vartheta$=0.3", fontsize=20)
ax3.set_title(L"L = 4, $\vartheta$=0.3", fontsize=20)
ax4.set_title("reference", fontsize=20)
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

##################### plot rank in energy #####################

fig = figure("rank in energy",figsize=(10, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = [L"rank $\mathbf{u}_{1}$",L"rank $\mathbf{u}_{c}$"]
for l = 1:L1
    ax.plot(rankInTime[1,1:(end-1)],rankInTime[l+1,1:(end-1)], ltype[l], linewidth=2, label=labelvec[l], alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("time", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper right", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure

fig = figure("rank in energy, ML",figsize=(10, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m:","g-.","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = [L"rank $\mathbf{u}_{1}$",L"rank $\mathbf{u}_{2}$",L"rank $\mathbf{u}_{3}$",L"rank $\mathbf{u}_{4}$",L"rank $\mathbf{u}_{c}$"]
for l = 1:L
    ax.plot(rankInTimeML[1,1:(end-1)],rankInTimeML[l+1,1:(end-1)], ltype[l], linewidth=2, label=labelvec[l], alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("time", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper right", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure