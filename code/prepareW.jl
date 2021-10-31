using Base: Float64
include("settings.jl")
include("SolverCSD.jl")

using PyCall
using PyPlot
using DelimitedFiles

close("all")

nx = 201;
s = Settings(nx,nx,50);
s.xMid = s.xMid./s.b
s.yMid = s.yMid./s.d
s.b = 1.0
s.d = 1.0
rhoMin = minimum(s.density);
## read density
density = Float64.(Gray.(load("liver_cut.jpg")))
nxD = size(density,1)
nyD = size(density,2)
y = collect(range(s.a,stop = s.b,length = nxD));
x = collect(range(s.c,stop = s.d,length = nyD));
############################
s.nPN = 21;
solver = SolverCSD(s);

W_dlr = readdlm("output/liverSmallPerson/W_csd_1stcollision_DLRA_Rank50nx200ny200nPN21eMax2.0rhoMin0.1.txt", Float64)

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

writedlm("output/W_plot_csd_1stcollision_DLRA_Rank$(s.r)nx$(s.NCellsX)ny$(s.NCellsY)nPN$(s.nPN)eMax$(s.eMax)rhoMin$(rhoMin).txt", O*W_modal)