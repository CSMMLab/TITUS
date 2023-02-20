using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using Trapz

close("all")

info = "CUDA"

nx = Int(floor(2 * 50));
ny = Int(floor(8 * 50));
nz = Int(floor(2 * 50));
problem ="validation" #"2DHighD"
particle = "Protons"
s = Settings(nx,ny,nz,5,problem, particle);

edges = (s.xMid, s.yMid, s.zMid,)
ray = (position=[s.x0,s.y0,s.z0], velocity=[0,1,0])
for hit in eachtraversal(ray, edges)
    println(hit)
end