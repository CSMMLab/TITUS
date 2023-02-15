function sph_cc(mu,phi,l,m)
    # Complex conjugates of coefficients.
    y = 0;
    z = computePlmx(mu,lmax=l,norm=SphericalHarmonics.Unnormalized())
    ma = abs(m);
    ind = Int(0.5*(l^2+l)+ma+1);
    
    y = y + sqrt((2*l+1)/(4*pi).*factorial(big(l-ma))./factorial(big(l+ma))).*(-1).^max(m,0).*exp(1im*m*phi).*z[ind];
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

function normpdf(x,mu,sigma)
    return 1/(sigma*sqrt(2*pi))*exp(-(x-mu)^2/2/(sigma^2));
end

function expm1div(x)
    # Function (exp(x)-1)/x that is accurate for x close to zero.
    y = 1+x*.5+x.^2/6;
    if abs(x)>2e-4;
        y = (exp(x)-1)./x;
    end
    return 1.0;#y;
end


function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*ny + j]
        end
    end
    return m;
end

function Vec2Mat(nx,ny,v::Array{Float64,2})
    n1 = size(v,2);
    m = zeros(nx,ny,n1);
    for i = 1:nx
        for j = 1:ny
            m[i,j,:] = v[(i-1)*ny + j,:]
        end
    end
    return m;
end

function Vec2Mat(nx,ny,v::Array{Float64,3})
    n1 = size(v,2);
    n2 = size(v,3);
    m = zeros(nx,ny,n1,n2);
    for i = 1:nx
        for j = 1:ny
            m[i,j,:,:] = v[(i-1)*ny + j,:,:]
        end
    end
    return m;
end

function expectedValue(v::Array{Float64,4})
    nx = size(v,1);
    ny = size(v,2);
    nMoments = size(v,3);
    nxi = size(v,4);
    m = zeros(nx,ny,nMoments);

    #xi, w = gausslegendre(nxi);
    xi = collect(range(0,1,nxi));
    w = 1.0/nxi*ones(size(xi))

    for i = 1:nx
        for j = 1:ny
            for l = 1:nMoments
                m[i,j,l] = sum(v[i,j,l,:] .* w)
            end
        end
    end
    return m;
end

function expectedValue(v::Array{Float64,3})
    nx = size(v,1);
    ny = size(v,2);
    nxi = size(v,3);
    m = zeros(nx,ny);

    #xi, w = gausslegendre(nxi);
    xi = collect(range(0,1,nxi));
    w = 1.0/nxi*ones(size(xi))

    for i = 1:nx
        for j = 1:ny
            m[i,j] = sum(v[i,j,:] .* w)
        end
    end
    return m;
end

function ExpVariance(v::Array{Float64,4})
    nx = size(v,1);
    ny = size(v,2);
    nMoments = size(v,3);
    nxi = size(v,4);
    Ev = zeros(nx,ny,nMoments);
    var = zeros(nx,ny,nMoments);

    #xi, w = gausslegendre(nxi);
    xi = collect(range(0,1,nxi));
    w = 1.0/nxi*ones(size(xi))

    for i = 1:nx
        for j = 1:ny
            for l = 1:nMoments
                Ev[i,j,l] = sum(v[i,j,l,:] .* w)
            end
        end
    end

    for l = 1:nxi
        var .+= w[l]*(v[:,:,:,l] .- Ev).^2;
    end

    return Ev, var;
end

function ExpVariance(v::Array{Float64,3})
    nx = size(v,1);
    ny = size(v,2);
    nxi = size(v,3);
    Ev = zeros(nx,ny);
    var = zeros(nx,ny);

    #xi, w = gausslegendre(nxi);
    xi = collect(range(0,1,nxi));
    w = 1.0/nxi*ones(size(xi))

    for i = 1:nx
        for j = 1:ny
            Ev[i,j] = sum(v[i,j,:] .* w)
        end
    end

    for l = 1:nxi
        var .+= w[l]*(v[:,:,l] .- Ev).^2;
    end

    return Ev, var;
end

function Mat2Vec(mat)
    nx = size(mat,1)
    ny = size(mat,2)
    m = size(mat,3)
    v = zeros(nx*ny,m);
    for i = 1:nx
        for j = 1:ny
            v[(i-1)*ny + j,:] = mat[i,j,:]
        end
    end
    return v;
end

function Ten2Ten(ten::Array{Float64,4}) # collapses order 4 to order 3 tensor
    nx = size(ten,1)
    ny = size(ten,2)
    m = size(ten,3)
    nxi = size(ten,4)
    v = zeros(nx*ny,m,nxi);
    for i = 1:nx
        for j = 1:ny
            v[(i-1)*ny + j,:,:] = ten[i,j,:,:]
        end
    end
    return v;
end

function Ten2Vec(ten)
    nx = size(ten,1)
    ny = size(ten,2)
    nxi = size(ten,3)
    m = size(ten,4)
    v = zeros(nx*ny,m*nxi);
    for i = 1:nx
        for j = 1:ny
            for l = 1:nxi
                for k = 1:m
                    v[(i-1)*ny + j,(l-1)*m .+ k] = ten[i,j,l,k]
                end
            end
        end
    end
    return v;
end

function vectorIndex(ny,i,j)
    return (i-1)*ny + j;
end

function FillMatrix(mat,r)
    if size(mat,2) != r
        tmp = mat;
        mat = zeros(size(tmp,1),r)
        for i = 1:size(tmp,2)
            mat[:,i] = tmp[:,i];
        end
    end
    return mat
end

function FillTensor(ten,r)
    if size(ten,1) != r || size(ten,2) != r || size(ten,3) != r
        tmp = ten;
        ten = zeros(r,r,r)
        for i = 1:size(tmp,1)
            for j = 1:size(tmp,2)
                for k = 1:size(tmp,3)
                    ten[i,j,k] = tmp[i,j,k];
                end
            end
        end
    end
    return ten
end

function FillTensor(ten,r1,r2,r3)
    if size(ten,1) != r1 || size(ten,2) != r2 || size(ten,3) != r3
        tmp = ten;
        ten = zeros(r1,r2,r3)
        for i = 1:size(tmp,1)
            for j = 1:size(tmp,2)
                for k = 1:size(tmp,3)
                    ten[i,j,k] = tmp[i,j,k];
                end
            end
        end
    end
    return ten
end

function Var(u::Array{Float64,3})
    nx = size(u,1);
    n = size(u,2);
    nxi = size(u,3);
    EU = zeros(nx,n);
    VarU = zeros(nx,n);
    xi, w = gausslegendre(nxi);
    for l = 1:nxi
        EU .+= w[l]*u[:,:,l]*0.5;
    end

    for l = 1:nxi
        VarU .+= 0.5*w[l]*(u[:,:,l] .- EU).^2;
    end
    return VarU;
end

function Var(u::Array{Float64,4})
    nx = size(u,1);
    ny = size(u,2);
    n = size(u,3);
    nxi = size(u,4);
    EU = zeros(nx,ny,n);
    VarU = zeros(nx,ny,n);
    xi, w = gausslegendre(nxi);
    for l = 1:nxi
        EU .+= w[l]*u[:,:,:,l]*0.5;
    end

    for l = 1:nxi
        VarU .+= 0.5*w[l]*(u[:,:,:,l] .- EU).^2;
    end
    return VarU;
end


function VarD(u::Array{Float64,3})
    nx = size(u,1);
    n = size(u,2);
    nxi = size(u,3);
    EU = zeros(nx,n);
    VarU = zeros(nx,n);
    w = 1/nxi * ones(nxi);
    for l = 1:nxi
        EU .+= w[l]*u[:,:,l];
    end

    for l = 1:nxi
        VarU .+= w[l]*(u[:,:,l] .- EU).^2;
    end
    return VarU;
end

function VarD(u::Array{Float64,4})
    nx = size(u,1);
    ny = size(u,2);
    n = size(u,3);
    nxi = size(u,4);
    EU = zeros(nx,ny,n);
    VarU = zeros(nx,ny,n);
    w = 1/nxi * ones(nxi);
    for l = 1:nxi
        EU .+= w[l]*u[:,:,:,l];
    end

    for l = 1:nxi
        VarU .+= w[l]*(u[:,:,:,l] .- EU).^2;
    end
    return VarU;
end