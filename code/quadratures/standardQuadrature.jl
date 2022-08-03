function createPointStandards(n::Int64)
    points,weights  = computeXYZandWeights(n)
	return points,weights
end


function computeXYZandWeights(_norder::Int64,resolve::String="y")
    n = _norder
    pointsxyz = zeros(Float64,2*n*n,3); # Even though we only need the x and y coordinate 
                                    # we still store the corresponding z coordinate
                                    # as our quadrature point actually lives on the sphere
    weights = zeros(n*n);

    # Construct Gauss quadrature
    #mu,gaussweights = gausslegendre(n)
    mu,gaussweights = gausslobatto(n)
        
    # around z axis equidistant
    phi = [(k+0.5)*pi/n for k=0:2*n-1]

    # make sure direction (1,0,0) lies in quadrature set. Note that _norder must be odd!
    if resolve == "x"
        phi .-= 0.5*pi/n;
    end

    # Transform between (mu,phi) and (x,y,z)
    x = sqrt.(1.0 .- mu.^2).*cos.(phi)'
    y = sqrt.(1.0 .- mu.^2).*sin.(phi)'
    z =           mu    .*ones(size(phi))'
    weights = 2.0*pi/n*repeat(gaussweights,1,2*n)
        
    # assign 
    pointsxyz[:,1] = x[:] 
    pointsxyz[:,2] = y[:]
    pointsxyz[:,3] = z[:]
        
    weights = weights[:]*0.5;
        
    
    return pointsxyz, weights
end

# this function thrwos away upper half of quadrature points
function computeXYZandWeightsProjected2D(_norder::Int64)
    n = _norder
    pointsxyz = zeros(Float64,n*n,3); # Even though we only need the x and y coordinate 
                                    # we still store the corresponding z coordinate
                                    # as our quadrature point actually lives on the sphere
    weights = zeros(n*n);

    # Construct Gauss quadrature
    mu,gaussweights = gausslegendre(n)
        
    # around z axis equidistant
    phi = [(k+0.5)*pi/n for k=0:2*n-1]

    range = 1:Int(ceil(n/2)) # we only use the upper half of the sphere as quadrature point since we do pseudo three d

    # Transform between (mu,phi) and (x,y,z)
    x = sqrt.(1.0 .- mu[range].^2).*cos.(phi)'
    y = sqrt.(1.0 .- mu[range].^2).*sin.(phi)'
    z =           mu[range]    .*ones(size(phi))'
    weights = 2.0*pi/n*repeat(gaussweights[range],1,2*n)
        
    # assign 
    pointsxyz[:,1] = x[:] 
    pointsxyz[:,2] = y[:]
    pointsxyz[:,3] = z[:]
        
    weights = weights[:]*0.5;
        
    
    return pointsxyz, weights
end

function computeXYZPlot(_norder::Int64)
    n = _norder
    pointsxyz = zeros(Float64,2*n*n,3); # Even though we only need the x and y coordinate 
                                    # we still store the corresponding z coordinate
                                    # as our quadrature point actually lives on the sphere

    # Construct Gauss quadrature
    mu,gaussweights = gausslegendre(n)
        
    # around z axis equidistant
    phi = [(k+0.5)*pi/n for k=0:2*n-1]

    range = 1:Int(ceil(n/1)) # we only use the upper half of the sphere as quadrature point since we do pseudo three d

    # Transform between (mu,phi) and (x,y,z)
    x = sqrt.(1.0 .- mu[range].^2).*cos.(phi)'
    y = sqrt.(1.0 .- mu[range].^2).*sin.(phi)'
    z =           mu[range]    .*ones(size(phi))'

    return x,y,z
end