using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using FastGaussQuadrature

close("all")

problem = "lungOrig"
nx = 31; ny = nx;
nxi = 30;

############################

s = Settings(nx,nx,nxi,25,problem);
solver = SolverCSD(s);
#X_dlr,W_dlr,U_dlr,C, dose_dlra,dose_dlra_var, psi_DLR,doseXi = SolveFirstCollisionSourceDLR(solver2);
u,dose_dlra,dose_dlra_var, psi_DLR = SolveFirstCollisionSourceUIOld(solver);
dose_dlra = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra);
dose_dlra_var = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra_var);

s = Settings(nx,nx,nxi,nxi,problem);
solver2 = SolverCSD(s);
xi = 0.0;
u, dose_full,VarDose, psi_full = SolveFirstCollisionSourceTensor(solver2);
dose_full = Vec2Mat(s.NCellsX,s.NCellsY,dose_full);
var_full = Vec2Mat(s.NCellsX,s.NCellsY,VarDose);


############################ Collocation ############################

s = Settings(nx,ny,nxi,5,problem);
rho0Inv = Mat2Vec(s.rho0Inv);
rho1Inv = Mat2Vec(s.rho1Inv);
xi, w = gausslegendre(nxi);
rhoFull = (rho0Inv*ones(nxi)' .+ rho1Inv*xi').^(-1);

doseXi = zeros(nxi,s.NCellsX,s.NCellsY);
solver2 = SolverCSD(s);
for k = 1:nxi
    solver2.dose .= zeros(size(solver2.dose))
    u, dose_full_tmp, psi_full_tmp = SolveFirstCollisionSource(solver2,rhoFull[:,k]);
    doseXi[k,:,:] = Vec2Mat(s.NCellsX,s.NCellsY,dose_full_tmp);
end

dose_full_col = zeros(s.NCellsX,s.NCellsY);
for l = 1:nxi
    dose_full_col .+= doseXi[l,:,:]/nxi;
end

var_full_col = zeros(size(dose_full_col));
# compute dose variance
for l = 1:nxi
    var_full_col .+= 1.0/nxi*(doseXi[l,:,:] .- dose_full_col).^2;
end

##################### plot dose contours #####################

## read density
density = Float64.(Gray.(load("LungOrig.png")))
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD-2));
x = collect(range(s.c,stop = s.d,length = nyD-2));
XX = (x'.*ones(size(y)))'
YY = (y'.*ones(size(x)))

# all contours magma
doseMax = maximum(dose_full[2:(end-1),2:(end-1)])
doseMaxVar = maximum(var_full[2:(end-1),2:(end-1)])
println("full = ",doseMax," ",doseMaxVar)
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,15),dpi=100)
ax1.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(Y,X,var_full[2:(end-1),2:(end-1)]'./doseMaxVar,levels,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("full", fontsize=20)
ax2.set_title("variance", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

# all contours magma collocation
doseMax = maximum(dose_full_col[2:(end-1),2:(end-1)])
doseMaxVar = maximum(var_full_col[2:(end-1),2:(end-1)])
println("full = ",doseMax," ",doseMaxVar)
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,15),dpi=100)
ax1.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(Y,X,dose_full_col[2:(end-1),2:(end-1)]'./doseMax,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(Y,X,var_full_col[2:(end-1),2:(end-1)]'./doseMaxVar,levels,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("collocation", fontsize=20)
ax2.set_title("collocation", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

# all contours magma DLRA
doseMax = maximum(dose_dlra[2:(end-1),2:(end-1)])
doseMaxVar = maximum(dose_dlra_var[2:(end-1),2:(end-1)])
println("DLRA = ",doseMax," ",doseMaxVar)
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,15),dpi=100)
ax1.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(Y,X,dose_dlra[2:(end-1),2:(end-1)]'./doseMax,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]'./doseMaxVar,levels,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("E[D]", fontsize=20)
ax2.set_title(L"Var[D]", fontsize=20)
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
doseMaxVar = maximum(var_full[2:(end-1),2:(end-1)])
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(Y,X,var_full[2:(end-1),2:(end-1)]'./doseMaxVar,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("E[D], full", fontsize=20)
ax2.set_title(L"Var[D], full", fontsize=20)
ax1.tick_params("both",labelsize=15) 
ax2.tick_params("both",labelsize=15) 
ax1.set_xlabel("x / [cm]", fontsize=15)
ax1.set_ylabel("y / [cm]", fontsize=15)
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax1.set_aspect(1)
ax2.set_aspect(1)
tight_layout()

# DLRA
doseMax = maximum(dose_dlra[2:(end-1),2:(end-1)])
doseMaxVar = maximum(dose_dlra_var[2:(end-1),2:(end-1)])
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,15),dpi=100)
CS = ax1.pcolormesh(Y,X,dose_dlra[2:(end-1),2:(end-1)]'./doseMax,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]'./doseMaxVar,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("E[D], DLRA", fontsize=20)
ax2.set_title(L"Var[D], DLRA", fontsize=20)
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
