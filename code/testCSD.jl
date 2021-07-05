include("settings.jl")
include("Solver.jl")
include("CSD.jl")
include("MaterialParameters.jl")

using PyPlot

s = Settings(100);

csd = CSD(s);
param = MaterialParameters();

# plot S at tabulated and finer data
fig, ax = subplots()
ax.plot(param.E_tab,param.S_tab, "k-", linewidth=2, label="S_tab", alpha=0.6)
ax.plot(csd.eGrid,csd.S, "r--o", linewidth=2, label="S", alpha=0.6)
ax.legend(loc="upper left")
ax.set_xlim([csd.eGrid[1],csd.eGrid[end]])
ax.tick_params("both",labelsize=20) 
show()

# test scattering at intermediate energy
indexE = 10
energy = param.E_tab[indexE]
sigma = SigmaAtEnergy(csd, energy)
sigmaRef = param.sigma_tab[indexE,:]
println("error sigma = ",norm(sigma.-sigmaRef))

# check intermediate energy
energy = 0.5*(param.E_tab[indexE]+param.E_tab[indexE+1])
sigma = SigmaAtEnergy(csd, energy)
sigmaRef = 0.5*(param.sigma_tab[indexE,:]+param.sigma_tab[indexE+1,:])
println("error sigma = ",norm(sigma.-sigmaRef))
