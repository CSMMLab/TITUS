include("utils.jl")
include("settings.jl")
include("CSD.jl")
include("MaterialParameters.jl")
include("MaterialParametersProtons.jl")

using PyPlot
using DelimitedFiles
using LegendrePolynomials

s = Settings();
s.particle = "Protons"

# plot proton cross sections

objCSD = CSD(s);
param = MaterialParametersProtons();

# plot proton cross sections
E = objCSD.eGrid;#[10.0 25.0 50.0] .+ s.eRest
nE = length(E)
xi = zeros(nE, s.nPN + 1)
for n = 1:nE
    xi[n,:] = SigmaAtEnergy(objCSD, E[n])
end

# plot stopping power
fig = figure("stopping power",figsize=(14,12),dpi=100)
ax = gca()
ax.plot(objCSD.eGrid[1:end-1] .- s.eRest,objCSD.S[1:end-1], linewidth=2, alpha=0.6)
ax.set_xscale("log")
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("E (MeV)",fontsize=15)
ax.set_ylabel(L"S",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
#savefig("output/StoppingPower.png")
#show()


param = MaterialParametersProtons();
E_tab = param.E_tab;
dOmega = 2.0;
OmegaC = 5;
OmegaF = 4.9;
Omega = [collect(range(dOmega,OmegaF,3)); collect(range(OmegaC,180 - OmegaC,170)); collect(range(180 - OmegaF,180 - dOmega,3))];
sigma_ce_mu, mu = ScreenedRutherfordAngle(E_tab,Omega)

N = 100;
xi = integrateXS_Poly(N,cosd.(Omega),E_tab,sigma_ce_mu)

# plot sigma at angles
close("all")
fig = figure("sigma at angles",figsize=(14,12),dpi=100)
ax = gca()
idx = [1]
lt = ["k-","r-","g-","b-","m-","y-"]
for l = 1:length(idx)
    ax.loglog(mu,sigma_ce_mu[l,:], lt[l], linewidth=2, label="E = $(E_tab[idx[l]])", alpha=0.6)
end
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel(L"\mu",fontsize=15)
ax.set_ylabel(L"\sigma",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
plt.show()
savefig("output/sigma_mu.png")

# plot sigma
fig = figure("xi",figsize=(14,12),dpi=100)
ax = gca()
idx = [1;2;3;4;5]
lt = ["k-","r-","g-","b-","m-","y-"]
for l = 1:length(idx)
    ax.loglog(E_tab,xi[:,idx[l]], lt[l], linewidth=2, label="order $(idx[l])", alpha=0.6)
end
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("E (MeV)",fontsize=15)
ax.set_ylabel(L"\xi",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
plt.show()
savefig("output/xi.png")

# plot sigma
fig = figure("xi order",figsize=(14,12),dpi=100)
ax = gca()
ax.loglog(collect(range(1,size(xi,2))),xi[1,:], linewidth=2, alpha=0.6)
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("order",fontsize=15)
ax.set_ylabel(L"\sigma",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
plt.show()
savefig("output/xi_order.png")