using Base: Float64
using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

include("settings.jl")
include("SolverCSD.jl")

close("all")

problem = "lungOrig"
nx = 201;
problem = "lung";#"waterBeam" #"2DHighD"
particle = "Electrons";#"Protons"

ϑᵤ = 0.01;
ϑₚ= 0.007;

############################

sᵤ = Settings(nx,nx,400,problem);
sᵤ.ϑ = ϑᵤ
solver1 = SolverCSD(sᵤ);
@time Xᵤ,Sᵤ,Wᵤ,_, doseᵤ, rankInTimeᵤ,ηᵤ,ηᵤBound, ψᵤ = SolveFirstCollisionSourceDLRBUGRejection(solver1);
doseᵤ = Vec2Mat(sᵤ.NCellsX,sᵤ.NCellsY,doseᵤ);

sₚ = Settings(nx,nx,400,problem);
sₚ.ϑ = ϑₚ
solverₚ = SolverCSD(sₚ);
@time Xₚ,Sₚ,Wₚ,_, doseₚ, rankInTimeₚ,ηₚ,ηₚBound,ψₚ = SolveFirstCollisionSourceDLRParallelRejection(solverₚ,false);
doseₚ = Vec2Mat(sₚ.NCellsX,sₚ.NCellsY,doseₚ);

s = Settings(nx,nx,100,problem,particle);
solver = SolverCSD(s);
@time u, dose_full, ψ_full = SolveFirstCollisionSource(solver);
dose_full = Vec2Mat(s.NCellsX,s.NCellsY,dose_full);

##################### plot dose #####################

## read density
density = Float64.(Gray.(load("Lung.png")))
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD-2));
x = collect(range(s.c,stop = s.d,length = nyD-2));
XX = (x'.*ones(size(y)))'
YY = (y'.*ones(size(x)))

# all contours magma
doseMax2 = maximum(doseᵤ[2:(end-1),2:(end-1)])
doseMax3 = maximum(doseₚ[2:(end-1),2:(end-1)])
doseMax4 = maximum(dose_full[2:(end-1),2:(end-1)])
levels = 40;
X = (sₚ.xMid[2:end-1]'.*ones(size(sₚ.yMid[2:end-1])))
Y = (sₚ.yMid[2:end-1]'.*ones(size(sₚ.xMid[2:end-1])))'


fig, ax = plt.subplots(1, 1,figsize=(8,8),dpi=100)
ax.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
ax.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax4,levels,cmap="plasma",vmin=0,vmax=1)
#ax.set_title(L"BUG, $\vartheta$="*LaTeXString(string(sᵤ.ϑ)), fontsize=20)
#ax.tick_params("both",labelsize=15) 
#ax.set_xlabel("x / [cm]", fontsize=15)
#ax.set_ylabel("y / [cm]", fontsize=15)
plt.axis("off")
ax.set_aspect(1)
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
#savefig("output/DoseIsolinesParallelTheta$(sᵤ.ϑ)BUGTheta$(sₚ.ϑ)nx$(nx)N$(s.nPN).png")

close("all")
XXd = (x'.*ones(size(y)))'
YYd = (y'.*ones(size(x)))
fig = figure("density",figsize=(10*(s.d/s.b),10),dpi=100)
ax = gca()
pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
#ax.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax4,levels,cmap="plasma",vmin=0,vmax=1)
#ax.contour(Y,X,doseᵤ[2:(end-1),2:(end-1)]'./doseMax2,levels,cmap="plasma",vmin=0,vmax=1)
ax.contour(Y,X,doseₚ[2:(end-1),2:(end-1)]'./doseMax3,levels,cmap="plasma",vmin=0,vmax=1)
ax.tick_params("both",labelsize=20) 
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
#plt.title("tissue density", fontsize=25)
plt.title("dose", fontsize=25)
tight_layout()

fig, (ax2, ax3) = plt.subplots(1, 2,figsize=(15,8),dpi=100)
ax2.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
ax2.contour(Y,X,doseᵤ[2:(end-1),2:(end-1)]'./doseMax2,levels,cmap="plasma",vmin=0,vmax=1)
ax3.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
CS = ax3.contour(Y,X,doseₚ[2:(end-1),2:(end-1)]'./doseMax3,levels,cmap="plasma",vmin=0,vmax=1)
ax2.set_title(L"BUG, $\vartheta$="*LaTeXString(string(sᵤ.ϑ)), fontsize=20)
ax3.set_title(L"parallel, $\vartheta$="*LaTeXString(string(sₚ.ϑ)), fontsize=20)
ax2.tick_params("both",labelsize=15) 
ax3.tick_params("both",labelsize=15) 
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax3.set_xlabel("x / [cm]", fontsize=15)
ax3.set_ylabel("y / [cm]", fontsize=15)
ax2.set_aspect(1)
ax3.set_aspect(1)
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/DoseIsolinesParallelTheta$(sᵤ.ϑ)BUGTheta$(sₚ.ϑ)nx$(nx)N$(s.nPN).png")


fig, (ax2, ax3) = plt.subplots(1, 2,figsize=(15,8),dpi=100)
ax2.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
ax3.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
CS = ax3.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax4,levels,cmap="plasma",vmin=0,vmax=1)
ax2.set_title("tissue density", fontsize=20)
ax3.set_title(L"P$_{21}$", fontsize=20)
ax2.tick_params("both",labelsize=15) 
ax3.tick_params("both",labelsize=15) 
ax2.set_xlabel("x / [cm]", fontsize=15)
ax2.set_ylabel("y / [cm]", fontsize=15)
ax3.set_xlabel("x / [cm]", fontsize=15)
ax3.set_ylabel("y / [cm]", fontsize=15)
ax2.set_aspect(1)
ax3.set_aspect(1)
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/DensityandDoseIsolinesReferencenx$(nx)N$(s.nPN).png")

fig, ax = plt.subplots(1, 1,figsize=(7.5,8),dpi=100)
ax.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
ax.set_title("tissue density", fontsize=20)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("x / [cm]", fontsize=15)
ax.set_ylabel("y / [cm]", fontsize=15)
ax.set_aspect(1)
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/tissuedensity.png")

fig, (ax) = plt.subplots(1, 1,figsize=(7.5,8),dpi=100)
ax.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
CS = ax.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax4,levels,cmap="plasma",vmin=0,vmax=1)
ax3.set_title(L"P$_{21}$", fontsize=20)
ax.tick_params("both",labelsize=15) 
ax.set_title(L"P$_{21}$", fontsize=20)
ax.set_xlabel("x / [cm]", fontsize=15)
ax.set_ylabel("y / [cm]", fontsize=15)
ax.set_aspect(1)
#cb = plt.colorbar(CS,fraction=0.035, pad=0.02)
#cb.ax.tick_params(labelsize=15)
tight_layout()
savefig("output/DoseIsolinesReferencenx$(nx)N$(s.nPN).png")

##################### plot rank in energy #####################

fig = figure("rank in energy",figsize=(14, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = ["BUG","parallel"]
#ax.plot(rankInTimeᵤ[1,1:(end-1)],rankInTimeᵤ[2,1:(end-1)], ltype[1], linewidth=2, label=labelvec[1], alpha=1.0)
#ax.plot(rankInTimeᵤ[1,1:(end-1)],rankInTimeₚ[2,1:(end-1)], ltype[2], linewidth=2, label=labelvec[2], alpha=1.0)
ax.plot(solver.csd.eGrid[2:end-2],rankInTimeᵤ[2,1:(end-1)], ltype[1], linewidth=2, label=labelvec[1], alpha=1.0)
ax.plot(solver.csd.eGrid[2:end-2],rankInTimeₚ[2,1:(end-1)], ltype[2], linewidth=2, label=labelvec[2], alpha=1.0)
ax.set_xlim([0.0,s.eMax])
ax.set_ylim([0.0,max(maximum(rankInTimeᵤ[2,2:end]),maximum(rankInTimeₚ[2,2:end])) + 1])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="lower left", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/ranksParallelTheta$(sᵤ.ϑ)BUGTheta$(sₚ.ϑ)nx$(nx)N$(s.nPN).png")

##################### plot eta in energy #####################

fig = figure("eta in energy",figsize=(14, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = ["BUG","parallel"]
ax.plot(solver.csd.eGrid[2:end-2],ηᵤ[:,2], "r-.", label=L"BUG, $\bar\vartheta =$ "*LaTeXString(string(ϑᵤ)), linewidth=2, alpha=1.0)
ax.plot(solver.csd.eGrid[2:end-2],ηₚ[:,2], "b--", label=L"parallel, $\bar\vartheta =$ "*LaTeXString(string(ϑₚ)), linewidth=2, alpha=1.0)
ax.plot(solver.csd.eGrid[2:end-2],ηᵤBound[:,2], "r:", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{BUG}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=1.0)
ax.plot(solver.csd.eGrid[2:end-2],ηₚBound[:,2], "b-", label=L"$c\Vert f\Vert\bar\vartheta_{\mathrm{parallel}} / h$, $c = $ "*LaTeXString(string(s.cη)), linewidth=2, alpha=0.4)
ax.set_xlim([0.0,s.eMax])
ax.set_ylim([0,max(maximum(ηᵤ[1:end,2]),maximum(ηₚ[1:end,2]),maximum(ηᵤBound[1:end,2]),maximum(ηₚBound[1:end,2]))+1])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel(L"\eta", fontsize=30);
ax.tick_params("both",labelsize=30);
ax.legend(loc="upper right", fontsize=30);
tight_layout()
fig.canvas.draw() # Update the figure
savefig("output/etaParallelTheta$(sᵤ.ϑ)BUGTheta$(sₚ.ϑ)nx$(nx)N$(s.nPN).png")

##################### dominant spatial modes #####################

fig, (ax1, ax3, ax2, ax4) = plt.subplots(2, 2,figsize=(15,13),dpi=100)
CS1 = ax1.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,Xᵤ[:,1])[2:(end-1),2:(end-1)]',cmap="viridis")
CS2 = ax2.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,Xₚ[:,1])[2:(end-1),2:(end-1)]',cmap="viridis")
CS3 = ax3.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,-Xᵤ[:,2])[2:(end-1),2:(end-1)]',cmap="viridis")
CS4 = ax4.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,-Xₚ[:,2])[2:(end-1),2:(end-1)]',cmap="viridis")
ax1.set_title(L"$X_1$, BUG", fontsize=20)
ax2.set_title(L"$X_1$, parallel", fontsize=20)
ax3.set_title(L"$X_2$, BUG", fontsize=20)
ax4.set_title(L"$X_2$, parallel", fontsize=20)
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
savefig("output/spatialModesTheta$(sᵤ.ϑ)BUGTheta$(sₚ.ϑ)nx$(nx)N$(s.nPN).png")

##################### dominant directional modes #####################

W_modal = solver.M*Wᵤ;

# setup quadrature
qorder = 100; # must be even for standard quadrature
qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
Q = Quadrature(qorder,qtype)

Ωs = Q.pointsxyz
weights = Q.weights
Norder = GlobalIndex( s.nPN, s.nPN ) + 1
nq = length(weights);

test = 1
counter = 1;
#Y = zeros(qorder +1,2*(qorder +1)+1,nq)
YY = zeros(Norder,nq)
@polyvar xx yy zz
count = 0;
for l=0:s.nPN
    for m=-l:l
        
        global counter
       
        sphericalh = ylm(l,m,xx,yy,zz)
        for q = 1 : nq
            YY[counter,q] =  sphericalh(xx=>Ωs[q,1],yy=>Ωs[q,2],zz=>Ωs[q,3])
        end
        global counter = counter+1;
    end
end
normY = zeros(Norder);
for i = 1:Norder
    for k = 1:nq
        normY[i] = normY[i] + YY[i,k]^2
    end
end
normY = sqrt.(normY)

Mat = zeros(nq,nq)
O = zeros(nq,Norder)
M = zeros(Norder,nq)
for i=1:nq
    local counter = 1
    for l=0:s.nPN
        for m=-l:l
            O[i,counter] = YY[counter,i] # / normY[counter] 
            M[counter,i] = YY[counter,i]*weights[i]# / normY[counter]
            counter += 1
        end
    end
end

writedlm("output/W_plot_lung.txt", O*W_modal)
writedlm("output/W_plot_lung_parallel.txt", O*solver.M*Wₚ)

run(`python plotWLungCompareParallel.py`)

rhoMin = minimum(s.density);
writedlm("output/dose1_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_full)
writedlm("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)theta$(sᵤ.ϑ).txt", doseᵤ)
writedlm("output/dose_csd_1stcollision_parallel_rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin)theta$(sₚ.ϑ).txt", doseₚ)

println("runLung finished")
