using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles

close("all")

function sph_cc(mu,phi,l,m)
    # Complex conjugates of coefficients.
    y = 0;
    z = computePlmx(mu,lmax=l,norm=SphericalHarmonics.Unnormalized())
    ma = abs(m);
    ind = Int(0.5*(l^2+l)+ma+1);
    
    y = y + sqrt((2*l+1)/(4*pi).*factorial(l-ma)./factorial(big(l+ma))).*(-1).^max(m,0).*exp(1im*m*phi).*z[ind];
    return y;
end

function real_sph(mu,phi,l,k)
    # Complex conjugates of coefficients.
    if k > 0
        return Float64((-1)^k/sqrt(2)*(sph_cc(mu,phi,l,k)+(-1)^k*sph_cc(mu,phi,l,-k)));
    elseif k < 0
        return Float64(-(-1)^k*1im/sqrt(2)*(sph_cc(mu,phi,l,-k)-(-1)^k*sph_cc(mu,phi,l,k)));
    else
        return Float64(sph_cc(mu,phi,l,k));
    end
end

nx = 151;
ny = 151;
nxi = 20;
problem = "lung"
s = Settings(nx,ny,nxi,50,problem);

solver = SolverCSD(s);

nTotalEntries = (s.nPN+1)^2;

# setup quadrature
Norder = GlobalIndex( s.nPN, s.nPN ) + 1
qorder = 100; # must be even for standard quadrature
qtype = 1; # Type must be 1 for "standard" or 2 for "octa" and 3 for "ico".
Q = Quadrature(qorder,qtype)
n = qorder;

# Construct Gauss quadrature
mu,gaussweights = gausslegendre(n)
    
# around z axis equidistant
phi = [(k+0.5)*pi/n for k=0:2*n-1]

# Transform between (mu,phi) and (x,y,z)
x = sqrt.(1.0 .- mu.^2).*cos.(phi)'
y = sqrt.(1.0 .- mu.^2).*sin.(phi)'
z =           mu    .*ones(size(phi))'
weights = 2.0*pi/n*repeat(gaussweights,1,2*n)
    
weights = weights[:]*0.5;

counter = 1;

nq = length(weights);

O = zeros(nq,Norder)
M = zeros(Norder,nq)
for l=0:solver.settings.nPN
    println(l)
    global counter;
    for m=-l:l
        for k = 1:length(mu)
            for j = 1:length(phi)
                O[(j-1)*n+k,counter] =  real_sph(mu[k],phi[j],l,m)
                M[counter,(j-1)*n+k] = O[(j-1)*n+k,counter]*weights[(j-1)*n+k]
            end
        end
        counter += 1
    end
end


w = solver.wXi;
global counter;
counter = 0;
for n = 1:1:130
    global counter;
    counter += 1;
    W_dlr = readdlm("output/gifW/W_$(n)", Float64)
    EW = zeros(nTotalEntries,s.r)
    stdW = zeros(nq,s.r)

    for l = 1:nxi
        EW .+= w[l]*W_dlr[(l-1)*nTotalEntries .+ (1:nTotalEntries),:];
    end

    for l = 1:nxi
        stdW .+= w[l]*(O*W_dlr[(l-1)*nTotalEntries .+ (1:nTotalEntries),:] .- O*EW).^2;
    end
    stdW .= sqrt.(stdW);

    if counter < 10
        name = "W_plot_00$(counter)"
    elseif counter < 100
        name = "W_plot_0$(counter)"
    else
        name = "W_plot_$(counter)"
    end

    writedlm("output/W_plot_lung.txt", [O*EW[:,1:2] stdW[:,1:2]])
    run(`python plotWLungE_std2.py "output/gifW_plot/$(name)"`)
end
