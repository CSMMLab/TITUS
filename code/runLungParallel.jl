using Base: Float64
using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK

include("settings.jl")
include("SolverCSD.jl")
include("SolverMLCSD.jl")

close("all")

problem = "lungOrig"
nx = 201;
problem = "lung";#"waterBeam" #"2DHighD"
particle = "Electrons";#"Protons"
s = Settings(nx,nx,100,problem,particle);

############################
solver = SolverCSD(s);
u, dose_full, psi_full = SolveFirstCollisionSource(solver);
dose_full = Vec2Mat(s.NCellsX,s.NCellsY,dose_full);

s = Settings(nx,nx,50,problem);
#s = Settings(nx,nx,int(maximum(rankInTime[2,:])));
solver2 = SolverCSD(s);
X_dlr,S_dlr,W_dlr,_,dose_dlra, psi_DLR = SolveFirstCollisionSourceDLR(solver2);
dose_dlra = Vec2Mat(s.NCellsX,s.NCellsY,dose_dlra);

L1 = 1;
s2 = Settings(nx,nx,400,problem);
s2.epsAdapt = 0.01
solver1 = SolverMLCSD(s2,L1);
X,S,W,_, dose_Llow, rankInTime, psi = SolveMCollisionSourceDLR(solver1);
dose_Llow = Vec2Mat(s2.NCellsX,s2.NCellsY,dose_Llow);

sp = Settings(nx,nx,400,problem);
sp.epsAdapt = 0.01
solverp = SolverCSD(sp);
X_p,S_dlrM,W_p,_, dose_p, rankInTime_p, psiML = SolveFirstCollisionSourceDLRParallel(solverp);
dose_P = Vec2Mat(sp.NCellsX,sp.NCellsY,dose_p);

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
doseMax1 = maximum(dose_dlra[2:(end-1),2:(end-1)])
doseMax2 = maximum(dose_Llow[2:(end-1),2:(end-1)])
doseMax3 = maximum(dose_P[2:(end-1),2:(end-1)])
doseMax4 = maximum(dose_full[2:(end-1),2:(end-1)])
levels = 40;
X = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
Y = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'

fig, (ax2, ax1, ax3, ax4) = plt.subplots(2, 2,figsize=(15,15),dpi=100)
ax1.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
CS = ax1.contour(Y,X,dose_dlra[2:(end-1),2:(end-1)]'./doseMax1,levels,cmap="plasma",vmin=0,vmax=1)
ax2.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
ax2.contour(Y,X,dose_Llow[2:(end-1),2:(end-1)]'./doseMax2,levels,cmap="plasma",vmin=0,vmax=1)
ax3.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
CS = ax3.contour(Y,X,dose_P[2:(end-1),2:(end-1)]'./doseMax3,levels,cmap="plasma",vmin=0,vmax=1)
ax4.pcolormesh(XX,YY,density[2:(end-1),2:(end-1)]',cmap="gray")
CS = ax4.contour(Y,X,dose_full[2:(end-1),2:(end-1)]'./doseMax4,levels,cmap="plasma",vmin=0,vmax=1)
ax1.set_title("fixed rank r = $(s.r)", fontsize=20)
ax2.set_title(L"Bug, $\bar{\vartheta}$=0.01", fontsize=20)
ax3.set_title(L"parallel, $\bar{\vartheta}$=0.01", fontsize=20)
ax4.set_title(L"P$_N$", fontsize=20)
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
labelvec = ["BUG","parallel"]
ax.plot(rankInTime[1,1:(end-1)],rankInTime[2,1:(end-1)], ltype[1], linewidth=2, label=labelvec[1], alpha=1.0)
ax.plot(rankInTime[1,1:(end-1)],rankInTime_p[2,1:(end-1)], ltype[2], linewidth=2, label=labelvec[2], alpha=1.0)
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper right", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure

fig = figure("rank in energy, ML",figsize=(10, 10), dpi=100)
ax = gca()
ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
labelvec = [L"rank $\mathbf{u}_{1}$",L"rank $\mathbf{u}_{c}$"]
for l = 1:L
    ax.plot(rankInTime_p[1,1:(end-1)],rankInTime_p[l+1,1:(end-1)], ltype[l], linewidth=2, label=labelvec[l], alpha=1.0)
end
ax.set_xlim([0.0,s.eMax])
#ax.set_ylim([0.0,440])
ax.set_xlabel("energy [MeV]", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
ax.legend(loc="upper right", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure

##################### dominant spatial modes #####################

fig, (ax1, ax3, ax2, ax4) = plt.subplots(2, 2,figsize=(15,13),dpi=100)
CS1 = ax1.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,1])[2:(end-1),2:(end-1)],cmap="plasma")
CS2 = ax2.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,2])[2:(end-1),2:(end-1)],cmap="plasma")
CS3 = ax3.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,3])[2:(end-1),2:(end-1)],cmap="plasma")
CS4 = ax4.pcolormesh(Y,X,Vec2Mat(s.NCellsX,s.NCellsY,X_dlr[:,4])[2:(end-1),2:(end-1)],cmap="plasma")
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

##################### dominant directional modes #####################

W_modal = solver.M*W_dlr;

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
println(counter)
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

run(`python plotWLung.py`)

rhoMin = minimum(s.density);
writedlm("output/dose1_csd_1stcollision_nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_full)
writedlm("output/dose_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_dlra)
writedlm("output/dose_csd_1stcollision_DLRAMlow_Rank$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_Llow)
writedlm("output/dose_csd_1stcollision_parallel$(s3.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", dose_P)

println("runLung finished")
