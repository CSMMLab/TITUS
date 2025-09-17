#ENV["JULIA_CUDA_USE_COMPAT"] = false #uncomment if there is problems with initialising CUDA related packages
#using MKL
using Base: Float64
T = Float32;
using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using Trapz
using TimerOutputs
using JLD2
using CUDA
#identify least used GPU and switch to that
devs = CUDA.devices()
mem_free = Float64[]

for (i,dev) in enumerate(devs)
    CUDA.device!(dev)

    # This forces context initialization
    CUDA.zeros(1)

    # Get free and total memory in bytes
    free, total = CUDA.memory_info()
    push!(mem_free, free)

    println("Device $(i -1): ", 
            " - Free memory: ", round(free / 1_048_576, digits=2), " MB / ",
            round(total / 1_048_576, digits=2), " MB")
end

# Select device with most free memory
best_idx = argmax(mem_free)
best_dev = collect(devs)[best_idx]

CUDA.device!(best_dev)
println("Selected GPU: $(best_idx - 1)")

using CSV
using DataFrames

include("settingsCSD.jl")
include("solverCSD.jl")

close("all")
    
info = "CUDA"
Nx = 20+3; Ny = 20+3; Nz =80+3;
nPN = 35;
r = 5;
epsAdapt = 0.1;
particle = "Protons"
problem = "matImport"
model = "Boltzmann"
order = 2;
gridScale = 1.0;

s = Settings(problem,model,Nx,Ny,Nz,nPN,r,epsAdapt,particle,order,gridScale);
rhoMin = minimum(s.density);
if CUDA.functional() 
    solver = solverCSD(s);
    solver.alpha = [0.0,0.0]
else
    println("GPU required to run this solver!")
end

time_elapsed=@elapsed dose_DLR = solve_rankAdaptive(solver,s.model);
println(time_elapsed)
dose_DLR = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_DLR);

idxX = Int(ceil(s.NCellsX/2))
idxY = Int(ceil(s.NCellsY/2))   #idxY = floor(Int,s.NCellsY/s.d*(0.5*s.d + s.y0))
idxZ = Int(ceil(s.NCellsZ/2))

X = (s.xMid'.*ones(size(s.yMid)))
Y = (s.yMid'.*ones(size(s.xMid)))'
Z = (s.zMid[3:end]'.*ones(size(s.yMid)))
XZ = (s.xMid'.*ones(size(s.zMid[3:end])))
ZX = (s.zMid[3:end]'.*ones(size(s.xMid)))
YZ = (s.yMid'.*ones(size(s.zMid[3:end])))'

#plot dose
fig = figure(dpi=100)
ax1 = gca()
im1 = ax1.pcolormesh(YZ',Z',dose_DLR[idxX,:,3:end]',vmin=0,vmax=maximum(dose_DLR),cmap="jet")
plt.colorbar(im1,ax=ax1)
ax1.title.set_text("Total deposited energy")
ax1.axis("equal")
show()
savefig("testCase_CSD/output/doseCSD_$(particle)_nPN$(s.nPN).png")

#plot dose cut
fig = figure(dpi=100)
ax = gca()
PyPlot.plot(s.zMid[3:end],dose_DLR[idxX,idxY,3:end]')
savefig("testCase_CSD/output/doseCSD_depthCut_$(particle)_nPN$(s.nPN).png")

