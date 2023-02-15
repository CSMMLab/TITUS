using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using FastGaussQuadrature

close("all")

problem = "timeCT"
nx = 50; ny = 50;
Nxi = 50;

############################ DLRA ############################

s = Settings(nx,ny,Nxi,5,problem);
solver = SolverCSD(s);
#X_dlr,W_dlr,U_dlr,C, dose_dlra,dose_dlra_var, psi_DLR,doseXi = SolveFirstCollisionSourceDLR(solver2);
X_dlr,W_dlr,U_dlr,C,dose_dlra,dose_dlra_var, psi_DLR,rankInTime = SolveFirstCollisionSourceUIAdaptive(solver);
dose_dlra = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra);
dose_dlra_var = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra_var);
u = ttm(C,[W_dlr],[2])[:,1,:]
u_dlra = Vec2Mat(s.NCellsX,s.NCellsY,ttm(u,[X_dlr,U_dlr],[1,2]));

EPhi,VarPhi = ExpVariance(u_dlra[:,:,:]);

doseMax = maximum(dose_dlra[2:(end-1),2:(end-1)]);
doseMaxVar = maximum(dose_dlra_var[2:(end-1),2:(end-1)]);
println("DLRA = ",doseMax," ",doseMaxVar)

############################ Collocation ############################

s = Settings(nx,ny,Nxi,5,problem);
rhoInvX = s.rhoInvX;
rhoInv = s.rhoInv;
rhoInvXi = s.rhoInvXi;
rhoFull = (rhoInvX*Diagonal(rhoInv)*rhoInvXi').^(-1);

doseXi = zeros(s.NCellsX,s.NCellsY,Nxi);
uXi = zeros(s.NCellsX,s.NCellsY,Nxi);
solver2 = SolverCSD(s);
for k = 1:Nxi
    solver2.dose .= zeros(size(solver2.dose))
    u, dose_full, psi_full = SolveFirstCollisionSource(solver2,rhoFull[:,k]);
    doseXi[:,:,k] = Vec2Mat(s.NCellsX,s.NCellsY,dose_full);
    uXi[:,:,k] = Vec2Mat(s.NCellsX,s.NCellsY,u[:,1]);
end

dose_full, var_dose_full = ExpVariance(doseXi);
EPhi_full, VarPhi_full = ExpVariance(uXi);


##################### plot Φ #####################

## read density
density = Float64.(Gray.(load("LungOrig.png")))
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD-2));
x = collect(range(s.c,stop = s.d,length = nyD-2));
XX = (x'.*ones(size(y)))'
YY = (y'.*ones(size(x)))

# full
doseMax = maximum(dose_full[2:(end-1),2:(end-1)])
doseMaxVar = maximum(var_dose_full[2:(end-1),2:(end-1)])
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

# compare expected value u0
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(Y,X,EPhi_full[2:(end-1),2:(end-1)]',cmap="plasma")
ax2.pcolormesh(Y,X,EPhi[2:(end-1),2:(end-1)]',cmap="plasma")
ax1.set_title(L"E$[\Phi]$, full", fontsize=20)
ax2.set_title(L"E$[\Phi]$, DLRA", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

# compare variance u0
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(Y,X,VarPhi_full[2:(end-1),2:(end-1)]',cmap="plasma")
ax2.pcolormesh(Y,X,VarPhi[2:(end-1),2:(end-1)]',cmap="plasma")
ax1.set_title(L"Var$[\Phi]$, full", fontsize=20)
ax2.set_title(L"Var$[\Phi]$, DLRA", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

##################### plot dose #####################

# compare expected value dose
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(Y,X,dose_full[2:(end-1),2:(end-1)]',cmap="plasma")
ax2.pcolormesh(Y,X,dose_dlra[2:(end-1),2:(end-1)]',cmap="plasma")
ax1.set_title(L"E$[D]$, full", fontsize=20)
ax2.set_title(L"E$[D]$, DLRA", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

# compare variance dose
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(Y,X,var_dose_full[2:(end-1),2:(end-1)]',cmap="plasma")
ax2.pcolormesh(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]',cmap="plasma")
ax1.set_title(L"Var$[D]$, full", fontsize=20)
ax2.set_title(L"Var$[D]$, DLRA", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

println("runLung finished")
