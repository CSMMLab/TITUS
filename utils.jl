function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
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

function Mat2Vec(m)
    nx = size(m,1);
    ny = size(m,2);
    v = zeros(nx*ny);
    for i = 1:nx
        for j = 1:ny
            v[(i-1)*ny + j] = m[i,j];
        end
    end
    return v;
end

## Extra functions
function rk(f, Y, h, order=1)
    if order == 1
        return Y .+ h*f(Y)
    elseif order == 2
        k1 = f(Y)
        k2 = f(Y + h * k1)
        return Y + 0.5 * h * (k1 + k2); # trapezoidal
        #return Y + h* f(Y + 0.5 * h * k1); # midpoint
    elseif order == 4
        k1 = f(Y)
        k2 = f(Y + 0.5*h*k1)
        k3 = f(Y + 0.5*h*k2)
        k4 = f(Y + h*k3)
        return Y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    else
        print("rk not implemented")
    end
end