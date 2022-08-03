function sph_cc(mu,phi,l,m)
    # Complex conjugates of coefficients.
    y = 0;
    z = computePlmx(mu,lmax=l,norm=SphericalHarmonics.Unnormalized())
    ma = abs(m);
    ind = Int(0.5*(l^2+l)+ma+1);
    
    y = y + sqrt((2*l+1)/(4*pi).*factorial(big(l-ma))./factorial(big(l+ma))).*(-1).^max(m,0).*exp(1im*m*phi).*z[ind];
    return y;
end

function sph_cc(mu,phi,l,m,z)
    # Complex conjugates of coefficients.
    ma = abs(m);
    ind = Int(0.5*(l^2+l)+ma+1);
    
    y = sqrt((2*l+1)/(4*pi).*factorial(big(l-ma))./factorial(big(l+ma))).*(-1).^max(m,0).*exp(1im*m*phi).*z[ind];
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

function real_sph(mu,phi,l,k,z)
    # Complex conjugates of coefficients.
    if k > 0
        return Float64((-1)^k/sqrt(2)*(sph_cc(mu,phi,l,k,z)+(-1)^k*sph_cc(mu,phi,l,-k,z)));
    elseif k < 0
        return Float64(-(-1)^k*1im/sqrt(2)*(sph_cc(mu,phi,l,-k,z)-(-1)^k*sph_cc(mu,phi,l,k,z)));
    else
        return Float64(sph_cc(mu,phi,l,k,z));
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