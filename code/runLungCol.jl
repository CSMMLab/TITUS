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
nx = 51; ny = 51;
Nxi = 100;

############################ DLRA ############################

s = Settings(nx,ny,Nxi,70,problem);
solver = SolverCSD(s);
#X_dlr,W_dlr,U_dlr,C, dose_dlra,dose_dlra_var, psi_DLR,doseXi = SolveFirstCollisionSourceDLR(solver2);
u,dose_dlra,dose_dlra_var, psi_DLR = SolveFirstCollisionSourceUI(solver);
dose_dlra = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra);
dose_dlra_var = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra_var);

doseMax = maximum(dose_dlra[2:(end-1),2:(end-1)]);
doseMaxVar = maximum(dose_dlra_var[2:(end-1),2:(end-1)]);
println("DLRA = ",doseMax," ",doseMaxVar)

############################ Collocation ############################

s = Settings(nx,ny,Nxi,5,problem);
rhoInvX = s.rhoInvX;
rhoInv = s.rhoInv;
rhoInvXi = s.rhoInvXi;
rhoFull = (rhoInvX*Diagonal(rhoInv)*rhoInvXi').^(-1);
#=rhoFullTestInv = zeros(size(rhoFull));
for k = 1:length(rhoInv)
    rhoFullTestInv .+= rhoInvX[:,k]*rhoInv[k]*rhoInvXi[:,k]';
end
rhoFullTest = rhoFullTestInv.^(-1);
println(norm(rhoFullTest - rhoFull))
=test rho full
XX = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
YY = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'
for i = 1:Nxi
    close("all")
    fig = figure("geo",figsize=(10*(s.d/s.b),10),dpi=100)
    ax = gca()
    pcolormesh(YY,XX,Vec2Mat(s.NCellsX,s.NCellsY,rhoFull[:,i])[2:end-1,2:end-1]',cmap="gray")
    ax.tick_params("both",labelsize=20) 
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.title("CT,  t = $i", fontsize=25)
    tight_layout()
    if i < 10
        savefig("CT_00$(i).png")
    elseif i < 100
        savefig("CT_0$(i).png")
    else
        savefig("CT_$(i).png")
    end
end=#

doseXi = zeros(Nxi,s.NCellsX,s.NCellsY);
solver2 = SolverCSD(s);
for k = 1:Nxi
    solver2.dose .= zeros(size(solver2.dose))
    u, dose_full, psi_full = SolveFirstCollisionSource(solver2,rhoFull[:,k]);
    doseXi[k,:,:] = Vec2Mat(s.NCellsX,s.NCellsY,dose_full);
end

dose_full = zeros(s.NCellsX,s.NCellsY);
for l = 1:Nxi
    dose_full .+= doseXi[l,:,:]/Nxi;
end

var_full = zeros(size(dose_full));
# compute dose variance
for l = 1:Nxi
    var_full .+= 1.0/Nxi*(doseXi[l,:,:] .- dose_full).^2;
end

##################### plot dose contours #####################
## read density
k = 1;
density = Float64.(Gray.(load("CTData/$(k)-070.png")));
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD-2));
x = collect(range(s.c,stop = s.d,length = nyD-2));
XX = (x'.*ones(size(y)))'
YY = (y'.*ones(size(x)))

# all contours magma
doseMax = maximum(dose_full[2:(end-1),2:(end-1)]);
doseMaxVar = maximum(var_full[2:(end-1),2:(end-1)]);
println("full = ",doseMax," ",doseMaxVar)
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18,10),dpi=100)
ax1.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(Y,X,var_full[2:(end-1),2:(end-1)]'./doseMaxVar,levels,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("full", fontsize=20)
ax2.set_title(L"L = 1, $\bar{\vartheta}$=0.01", fontsize=20)
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
doseMax = maximum(dose_dlra[2:(end-1),2:(end-1)]);
doseMaxVar = maximum(dose_dlra_var[2:(end-1),2:(end-1)]);
println("DLRA = ",doseMax," ",doseMaxVar)
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18,10),dpi=100)
ax1.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
CS = ax1.contour(Y,X,dose_dlra[2:(end-1),2:(end-1)]'./doseMax,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(XX',YY',density[2:(end-1),2:(end-1)],cmap="gray")
ax2.contour(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]'./doseMaxVar,4*levels,cmap="plasma",vmin=0,vmax=1)
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

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18,10),dpi=100)
#CS = ax1.pcolormesh(Y,X,dose_dlra[2:(end-1),2:(end-1)]'./doseMax,cmap="plasma",vmin=0,vmax=1)
#ax2.pcolormesh(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]'./doseMaxVar,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(Y,X,log.(var_full[2:(end-1),2:(end-1)]'),cmap="plasma")
ax1.pcolormesh(Y,X,var_full[2:(end-1),2:(end-1)]'./doseMaxVar,cmap="plasma")
ax1.set_title("Var[D], full", fontsize=20)
ax2.set_title("Var[D], full log", fontsize=20)
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

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18,10),dpi=100)
#CS = ax1.pcolormesh(Y,X,dose_dlra[2:(end-1),2:(end-1)]'./doseMax,cmap="plasma",vmin=0,vmax=1)
#ax2.pcolormesh(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]'./doseMaxVar,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(Y,X,log.(dose_dlra_var[2:(end-1),2:(end-1)]'),cmap="plasma")
ax1.pcolormesh(Y,X,dose_dlra_var[2:(end-1),2:(end-1)]'./doseMaxVar,cmap="plasma")
ax1.set_title("Var[D], DLRA", fontsize=20)
ax2.set_title("Var[D], DLRA log", fontsize=20)
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
