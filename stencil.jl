__precompile__

using ProgressMeter
using LinearAlgebra
using LegendrePolynomials
using QuadGK
using SparseArrays
using SphericalHarmonicExpansions, SphericalHarmonics, TypedPolynomials, GSL
using MultivariatePolynomials
using PyCall
np = pyimport("numpy")

include("utils.jl")

struct Stencil
    Dxx::SparseMatrixCSC{Float64,Int64}
    Dyy::SparseMatrixCSC{Float64,Int64}
    Dx::SparseMatrixCSC{Float64,Int64}
    Dy::SparseMatrixCSC{Float64,Int64}

    # constructor
    function Stencil(settings, nx, ny)
        # setup stencil matrix
        Dxx = spzeros(nx * ny, nx * ny)
        Dyy = spzeros(nx * ny, nx * ny)
        Dx = spzeros(nx * ny, nx * ny)
        Dy = spzeros(nx * ny, nx * ny)

        Δy = settings.Δx
        Δx = settings.Δx

        # setup index arrays and values for allocation of stencil matrices
        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3 * (nx - 2) * (ny - 2))
        J = zeros(3 * (nx - 2) * (ny - 2))
        vals = zeros(3 * (nx - 2) * (ny - 2))
        counter = -2

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3
                # x part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i + 1, j)
                indexMinus = vectorIndex(nx, i - 1, j)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 2.0 / 2 / Δx
                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / Δx
                end
                if i < nx
                    II[counter+2] = index
                    J[counter+2] = indexPlus
                    vals[counter+2] = -1 / 2 / Δx
                end
            end
        end
        Dxx = sparse(II, J, vals, nx * ny, nx * ny)

        II .= zeros(3 * (nx - 2) * (ny - 2))
        J .= zeros(3 * (nx - 2) * (ny - 2))
        vals .= zeros(3 * (nx - 2) * (ny - 2))
        counter = -2

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3
                # y part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i, j + 1)
                indexMinus = vectorIndex(nx, i, j - 1)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 2.0 / 2 / Δy

                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / Δy
                end
                if j < ny
                    II[counter+2] = index
                    J[counter+2] = indexPlus
                    vals[counter+2] = -1 / 2 / Δy
                end
            end
        end
        Dyy = sparse(II, J, vals, nx * ny, nx * ny)

        II = zeros(2 * (nx - 2) * (ny - 2))
        J = zeros(2 * (nx - 2) * (ny - 2))
        vals = zeros(2 * (nx - 2) * (ny - 2))
        counter = -1

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2
                # x part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i + 1, j)
                indexMinus = vectorIndex(nx, i - 1, j)

                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / Δx
                end
                if i < nx
                    II[counter+1] = index
                    J[counter+1] = indexPlus
                    vals[counter+1] = 1 / 2 / Δx
                end
            end
        end
        Dx = sparse(II, J, vals, nx * ny, nx * ny)

        II .= zeros(2 * (nx - 2) * (ny - 2))
        J .= zeros(2 * (nx - 2) * (ny - 2))
        vals .= zeros(2 * (nx - 2) * (ny - 2))
        counter = -1

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2
                # y part
                index = vectorIndex(nx, i, j)
                indexPlus = vectorIndex(nx, i, j + 1)
                indexMinus = vectorIndex(nx, i, j - 1)

                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / 2 / Δy
                end
                if j < ny
                    II[counter+1] = index
                    J[counter+1] = indexPlus
                    vals[counter+1] = 1 / 2 / Δy
                end
            end
        end
        Dy = sparse(II, J, vals, nx * ny, nx * ny)
        
        new(Dxx, Dyy, Dx, Dy)
    end
end