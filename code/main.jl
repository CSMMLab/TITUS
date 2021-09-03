using Base: Float64
include("settings.jl")
include("SolverCSD.jl")
include("driver_dlr.jl")

using PyCall
using PyPlot
using DelimitedFiles

close("all")

s = Settings(501); #WaterPhantomKerstin Nx = 5001, 

if s.problem == "AirCavity"
    smapIn = readdlm("dose_ac.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
elseif s.problem == "WaterPhantomKerstin"
    smapIn = readdlm("doseStarmapWaterPhantom.txt", ',', Float64)
    xRef = smapIn[:,1]
    doseRef = smapIn[:,2]
else
    xRef = 0; doseRef = 1;
end

############################
solver = SolverCSD(s)

@time X,S,W, = SolveNaiveUnconventional(solver);
u_2 = X*S*W';
u_2=0.5*sqrt(solver.gamma[1])*u_2;

@time u_1, dose = Solve(solver);

fig, ax = subplots()
ax.plot(s.xMid,u_1[:,1], "r--", linewidth=2, label="PN", alpha=0.6)
ax.plot(s.xMid,u_2[:,1], "b--", linewidth=2, label="PN DLR", alpha=0.6)
ax.legend(loc="upper left")
ax.set_xlim([s.a,s.b])
ax.tick_params("both",labelsize=20) 
fig.savefig("scalarFlux.png", dpi=fig.dpi)

# fig, ax = subplots()
# # maxDose = maximum(dose[Integer(floor(s.NCells*(0.2-s.a)/(s.b-s.a))):end]); # maximum dose value starting at x = 0.5
# # ax.plot(s.xMid,dose./maxDose, "r--", linewidth=2, label="PN dose", alpha=0.8)
# #ax.plot(s.xMid,s.density, "k-", linewidth=2, label="Starmap dose", alpha=0.6)
# # ax.plot(xRef.-2.0,doseRef./maximum(doseRef), "k-", linewidth=2, label="Starmap dose", alpha=0.6)
# ax.legend(loc="upper left")
# ax.set_xlim([0,s.b])
# #ax.set_ylim([0,1])
# ax.tick_params("both",labelsize=20) 
# fig.savefig("dose.png", dpi=fig.dpi)

println("main finished")
