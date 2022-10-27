include("utils.jl")
include("settings.jl")
include("CSD.jl")
include("MaterialParameters.jl")
include("MaterialParametersProtons.jl")

using PyPlot
using DelimitedFiles
using LegendrePolynomials

s = Settings();

# plot electron cross sections

csd = CSD(s);
param = MaterialParameters();

muOrig = readdlm("validationData/muGrid.txt")
mu2 = [-muOrig[end:-1:2];muOrig]
mu = 1 .-2*muOrig;
#1-2*muorig = ( 1.0e0 - std::cos( THR[i] ) ) 
E = readdlm("validationData/eGrid.txt").*1e-6;
sigma = readdlm("validationData/scatteringData.txt")
#sigma = [sigma[:,end:-1:2] sigma[:,:]]
xiRef = param.sigma_tab;
E_tab = param.E_sigmaTab;

N = 40;
nMu = length(mu);
nE = length(E);
P = zeros(N,nMu)
for k = 1:nMu
    Ptmp = collectPl(mu[k], lmax = N-1);
    for i = 1:N
        P[i,k] = Ptmp[i-1]
    end
end

xi = zeros(nE,N)
for n = 1:nE
    for i = 1:N
        for k = 1:nMu-1
            xi[n,i] -= 0.5*(mu[k+1]-mu[k])*(sigma[n,k]*P[i,k] + sigma[n,k+1]*P[i,k+1]);
        end
    end
end

# plot sigma
fig, ax = subplots()
ax.plot(mu,sigma[5,:], "k-", linewidth=2, label="S_tab", alpha=0.6)
ax.legend(loc="upper left")
ax.tick_params("both",labelsize=20) 
show()

# plot sigma
fig = figure("sigma",figsize=(14,12),dpi=100)
ax = gca()
idx = [1;2;3;4;5]
lt = ["k-","r-","g-","b-","m-","y-"]
ltS = ["k:","r:","g:","b:","m:","y:"]
for l = 1:length(idx)
    ax.loglog(E,xi[:,idx[l]], lt[l], linewidth=2, label="order $(idx[l])", alpha=0.6)
    ax.loglog(E_tab,xiRef[:,idx[l]], ltS[l], linewidth=2, label="order $(idx[l]), Starmap", alpha=0.6)
end
ax.legend(loc="upper right",fontsize=15)
ax.tick_params("both",labelsize=15) 
ax.set_xlabel("E (MeV)",fontsize=15)
ax.set_ylabel(L"\sigma",fontsize=15)
PyPlot.grid(true, which="both")
tight_layout();
show()
