include("settings.jl")
include("PNSystem.jl")

using PyPlot

s = Settings(100);

pn = PNSystem(s)
SetupSystemMatrices(pn);
