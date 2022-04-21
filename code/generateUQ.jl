using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using FastGaussQuadrature
using LinearAlgebra
using Interpolations

include("settings.jl")
include("utils.jl")

nx = 300;
ny = 300;
nxi = 20;
problem = "lung"
s = Settings(nx+1,ny+1,nxi,20,problem);
densityTmp = zeros(nx,ny);
ndata = 10;
densityInv = zeros(nx*ny,ndata);
for k = 1:ndata
    img = Float64.(Gray.(load("CTData/$(k)-070.png")));
    nxi = size(img,1);
    nyi = size(img,2);
    densityMin = 0.05;
    for i = 1:nx
        for j = 1:ny
            densityTmp[i,j] = max(1.85*img[max(Int(floor(i/nx*nxi)),1),max(Int(floor(j/ny*nyi)),1)],densityMin) # 1.85 bone, 1.04 muscle, 0.3 lung
        end
    end
    densityInv[:,k] = 1.0./Mat2Vec(densityTmp);
end

nf = 100;
xi_tab = collect(range(0,1,10));
xi = collect(range(0,1,nf));
densityInvF = zeros(nx*ny,nf);

for j = 1:nx*ny
    xiToDensity = LinearInterpolation(xi_tab, densityInv[j,:]; extrapolation_bc=Throw())
    for i = 1:nf  
        densityInvF[j,i] = xiToDensity(xi[i])
    end
end

U,S,V = svd(densityInvF)

XX = (s.xMid[2:end-1]'.*ones(size(s.yMid[2:end-1])))
YY = (s.yMid[2:end-1]'.*ones(size(s.xMid[2:end-1])))'
for i = 1:nf
    close("all")
    fig = figure("geo",figsize=(10*(s.d/s.b),10),dpi=100)
    ax = gca()
    pcolormesh(YY,XX,Vec2Mat(s.NCellsX,s.NCellsY,1.0./densityInvF[:,i])[2:end-1,2:end-1]',cmap="gray")
    ax.tick_params("both",labelsize=20) 
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.title("CT,  t = $(round(xi[i], digits=3))", fontsize=25)
    tight_layout()
    if i < 10
        savefig("output/gifgeo/CT_00$(i).png")
    elseif i < 100
        savefig("output/gifgeo/CT_0$(i).png")
    else
        savefig("output/gifgeo/CT_$(i).png")
    end
end

