# Generic helpers for UQ test cases with the 2D radiation tree ((space, angle), (ξ, η)):
# source compression into a TTN, full-tensor evaluation and reference mode products, quadrature
# for the random parameters, and statistics

using LinearAlgebra
using FastGaussQuadrature
using TensorToolbox

# boundary (ghost) cells are not updated by the stencils and must not carry a source
function interiorMask(s::Settings)
    nx = s.NCells; ny = nx
    m = zeros(nx * ny)
    for i = 2:nx-1, j = 2:ny-1
        m[vectorIndex(ny, i, j)] = 1.0
    end
    return m
end

# truncated left singular vectors such that the discarded part has relative norm ≤ tol
function truncatedBasis(M::AbstractMatrix, tol::Float64)
    U, σ, _ = svd(M)
    tail = sqrt.(reverse(cumsum(reverse(σ .^ 2))))   # tail[q] = ‖σ[q:end]‖
    q = findfirst(t -> t <= tol * norm(σ), tail)
    q = q === nothing ? length(σ) : max(q - 1, 1)
    return U[:, 1:q]
end

# Build the source TTN 
function generateSourceTTN(s::Settings, Q₀₀::Array{Float64,3}; tol::Float64=1e-6, angle=nothing)
    n, nξ, nη = size(Q₀₀)
    nΩ = GlobalIndex(s.nPN, s.nPN) + 1
    if angle !== nothing
        Q₀₀ = norm(angle) .* Q₀₀
    end

    Uₓ = truncatedBasis(reshape(Q₀₀, n, nξ * nη), tol)
    Uξ = truncatedBasis(reshape(permutedims(Q₀₀, (2, 1, 3)), nξ, n * nη), tol)
    Uη = truncatedBasis(reshape(permutedims(Q₀₀, (3, 1, 2)), nη, n * nξ), tol)
    UΩ = zeros(nΩ, 1); UΩ[1, 1] = 1.0
    if angle !== nothing
        UΩ[:, 1] .= angle ./ norm(angle)
    end

    G = ttm(Q₀₀, [Matrix(Uₓ'), Matrix(Uξ'), Matrix(Uη')], [1, 2, 3])   # qx × qξ × qη
    qx, qξ, qη = size(Uₓ, 2), size(Uξ, 2), size(Uη, 2)

    # split (space, angle) | (ξ, η) with a truncated SVD of the reduced core
    U₁, σ, V₁ = svd(reshape(G, qx, qξ * qη))
    tail = sqrt.(reverse(cumsum(reverse(σ .^ 2))))
    q = findfirst(t -> t <= tol * norm(σ), tail)
    q = q === nothing ? length(σ) : max(q - 1, 1)
    U₁ = U₁[:, 1:q]; V₁ = V₁[:, 1:q]; σ = σ[1:q]

    subleaf1 = TTN(1, Matrix(Uₓ))   # space
    subleaf2 = TTN(2, UΩ)           # angle (moments)
    subleaf3 = TTN(3, Matrix(Uξ))   # ξ
    subleaf4 = TTN(4, Matrix(Uη))   # η
    leaf1 = TTN(3, matten(Matrix(U₁'), 1, [q, qx, 1]))
    leaf2 = TTN(2, matten(Matrix(V₁'), 1, [q, qξ, qη]))
    root = TTN(1, reshape(Matrix(Diagonal(σ)), 1, q, q))
    root.S = ones(Float64, 1, 1)
    root.VᵀFV = ones(Float64, 1, 1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# full tensor (n × nΩ × nξ × nη) represented by a TTN with the 2D radiation tree
function fullTensor(Y::TTN, s::Settings)
    n = s.NCells^2
    nΩ = GlobalIndex(s.nPN, s.nPN) + 1
    return reshape(eval(Y, Any[:, :, :, :]), n, nΩ, s.Nxi, s.Neta)
end

# mode-m product U ×ₘ A for a 4-tensor; identities are skipped, diagonals are broadcast
function modeprod(U::Array{Float64,4}, A, m::Int)
    if A isa Diagonal
        all(A.diag .== 1.0) && return U
        shp = ones(Int, 4); shp[m] = length(A.diag)
        return U .* reshape(A.diag, shp...)
    end
    perm = [m; setdiff(1:4, m)]
    Up = permutedims(U, perm)
    sz = size(Up)
    V = reshape(A * reshape(Up, sz[1], :), size(A, 1), sz[2:end]...)
    return permutedims(V, invperm(perm))
end

# quadrature nodes and probability weights (summing to one) for a random parameter
# :gaussian -> standard normal N(0,1) (Gauss-Hermite), :uniform -> U[-1,1] (Gauss-Legendre)
function parameterQuadrature(n::Int, distribution::Symbol)
    if distribution == :gaussian
        x, w = gausshermite(n)              # weight exp(-x²)
        return sqrt(2) .* x, w ./ sqrt(π)
    elseif distribution == :uniform
        x, w = gausslegendre(n)
        return x, w ./ 2
    else
        error("unknown distribution $distribution")
    end
end

# mean and variance over (ξ, η) of a field F (ncells × nξ × nη) for probability weights pξ, pη
function uqStatistics(F::Array{Float64,3}, pξ, pη)
    w = pξ * pη'
    E = [sum(w .* F[j, :, :]) for j in axes(F, 1)]
    V = [sum(w .* (F[j, :, :] .- E[j]) .^ 2) for j in axes(F, 1)]
    return E, V
end
