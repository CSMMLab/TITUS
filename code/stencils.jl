__precompile__

using SparseArrays

include("utils.jl")

struct Stencils{T<:AbstractFloat}
    L1x::SparseMatrixCSC{T, Int64};
    L1y::SparseMatrixCSC{T, Int64};
    L1z::SparseMatrixCSC{T, Int64};
    L2x::SparseMatrixCSC{T, Int64};
    L2y::SparseMatrixCSC{T, Int64};
    L2z::SparseMatrixCSC{T, Int64};
    function Stencils(settings::Settings,T::DataType=Float64,order::Int=2)
        density = settings.density;
        # setup stencil matrix
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        nz = settings.NCellsZ;
        L1x = spzeros(nx*ny*nz,nx*ny*nz);
        L1y = spzeros(nx*ny*nz,nx*ny*nz);
        L1z = spzeros(nx*ny*nz,nx*ny*nz);
        L2x = spzeros(nx*ny*nz,nx*ny*nz);
        L2y = spzeros(nx*ny*nz,nx*ny*nz);
        L2z = spzeros(nx*ny*nz,nx*ny*nz);

        if order == 2
            # setup index arrays and values for allocation of stencil matrices
            II = zeros(3*(nx-2)*(ny-2)*(nz-2)); J = zeros(3*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,3*(nx-2)*(ny-2)*(nz-2));
            counter = -2;

            for i = 2:nx-1
                for j = 2:ny-1
                    for k = 2:nz-1
                        counter = counter + 3;
                        # x part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i+1,j,k);
                        indexMinus = vectorIndex(nx,ny,i-1,j,k);

                        II[counter+1] = index;
                        J[counter+1] = index;
                        vals[counter+1] = 2.0/2/settings.dx/density[i,j,k]; 
                        if i > 1
                            II[counter] = index;
                            J[counter] = indexMinus;
                            vals[counter] = -1/2/settings.dx/density[i-1,j,k];
                        end
                        if i < nx
                            II[counter+2] = index;
                            J[counter+2] = indexPlus;
                            vals[counter+2] = -1/2/settings.dx/density[i+1,j,k]; 
                        end
                    end
                end
            end
            L1x = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(3*(nx-2)*(ny-2)*(nz-2)); J = zeros(3*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,3*(nx-2)*(ny-2)*(nz-2));
            counter = -2;

            for i = 2:nx-1
                for j = 2:ny-1
                    for k = 2:nz-1
                        counter = counter + 3;
                        # y part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i,j+1,k);
                        indexMinus = vectorIndex(nx,ny,i,j-1,k);

                        II[counter+1] = index;
                        J[counter+1] = index;
                        vals[counter+1] = 2.0/2/settings.dy/density[i,j,k]; 

                        if j > 1
                            II[counter] = index;
                            J[counter] = indexMinus;
                            vals[counter] = -1/2/settings.dy/density[i,j-1,k];
                        end
                        if j < ny
                            II[counter+2] = index;
                            J[counter+2] = indexPlus;
                            vals[counter+2] = -1/2/settings.dy/density[i,j+1,k]; 
                        end
                    end
                end
            end
            L1y = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(3*(nx-2)*(ny-2)*(nz-2)); J = zeros(3*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,3*(nx-2)*(ny-2)*(nz-2));
            counter = -2;

            for i = 2:nx-1
                for j = 2:ny-1
                    for k = 2:nz-1
                        counter = counter + 3;
                        # y part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i,j,k+1);
                        indexMinus = vectorIndex(nx,ny,i,j,k-1);

                        II[counter+1] = index;
                        J[counter+1] = index;
                        vals[counter+1] = 2.0/2/settings.dz/density[i,j,k]; 

                        if k > 1
                            II[counter] = index;
                            J[counter] = indexMinus;
                            vals[counter] = -1/2/settings.dz/density[i,j,k-1];
                        end
                        if k < nz
                            II[counter+2] = index;
                            J[counter+2] = indexPlus;
                            vals[counter+2] = -1/2/settings.dz/density[i,j,k+1]; 
                        end
                    end
                end
            end
            L1z = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,2*(nx-2)*(ny-2)*(nz-2));
            counter = -1;

            for i = 2:nx-1
                for j = 2:ny-1
                    for k = 2:nz-1
                        counter = counter + 2;
                        # x part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i+1,j,k);
                        indexMinus = vectorIndex(nx,ny,i-1,j,k);

                        if i > 1
                            II[counter] = index;
                            J[counter] = indexMinus;
                            vals[counter] = -1/2/settings.dx/density[i-1,j,k];
                        end
                        if i < nx
                            II[counter+1] = index;
                            J[counter+1] = indexPlus;
                            vals[counter+1] = 1/2/settings.dx/density[i+1,j,k];
                        end
                    end
                end
            end
            L2x = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,2*(nx-2)*(ny-2)*(nz-2));
            counter = -1;

            for i = 2:nx-1
                for j = 2:ny-1
                    for k = 2:nz-1
                        counter = counter + 2;
                        # y part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i,j+1,k);
                        indexMinus = vectorIndex(nx,ny,i,j-1,k);

                        if j > 1
                            II[counter] = index;
                            J[counter] = indexMinus;
                            vals[counter] = -1/2/settings.dy/density[i,j-1,k];
                        end
                        if j < ny
                            II[counter+1] = index;
                            J[counter+1] = indexPlus;
                            vals[counter+1] = 1/2/settings.dy/density[i,j+1,k];
                        end
                    end
                end
            end
            L2y = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,2*(nx-2)*(ny-2)*(nz-2));
            counter = -1;

            for i = 2:nx-1
                for j = 2:ny-1
                    for k = 2:nz-1
                        counter = counter + 2;
                        # y part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i,j,k+1);
                        indexMinus = vectorIndex(nx,ny,i,j,k-1);

                        if k > 1
                            II[counter] = index;
                            J[counter] = indexMinus;
                            vals[counter] = -1/2/settings.dz/density[i,j,k-1];
                        end
                        if k < nz
                            II[counter+1] = index;
                            J[counter+1] = indexPlus;
                            vals[counter+1] = 1/2/settings.dz/density[i,j,k+1];
                        end
                    end
                end
            end
            L2z = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);
        elseif order == 4
            # setup index arrays and values for allocation of stencil matrices
            II = zeros(4*(nx-4)*(ny-4)*(nz-4)); J = zeros(4*(nx-4)*(ny-4)*(nz-4)); vals = zeros(T,4*(nx-4)*(ny-4)*(nz-4));
            counter = -3;

            for i = 3:nx-2
                for j = 3:ny-2
                    for k = 3:nz-2
                        counter = counter + 4;
                        # x part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i+1,j,k);
                        indexMinus = vectorIndex(nx,ny,i-1,j,k);
                        indexPP = vectorIndex(nx,ny,i+2,j,k);
                        indexMM = vectorIndex(nx,ny,i-2,j,k);

                        if i > 2
                            II[counter] = index;
                            J[counter] = indexMM;
                            vals[counter] = 1/12/settings.dx/density[i-2,j,k];
                        end
                        if i > 1
                            II[counter+1] = index;
                            J[counter+1] = indexMinus;
                            vals[counter+1] = -8/12/settings.dx/density[i-1,j,k];
                        end
                        if i < nx
                            II[counter+2] = index;
                            J[counter+2] = indexPlus;
                            vals[counter+2] = +8/12/settings.dx/density[i+1,j,k]; 
                        end
                        if i < nx-1
                            II[counter+3] = index;
                            J[counter+3] = indexPP;
                            vals[counter+3] = -1/12/settings.dx/density[i+2,j,k]; 
                        end
                    end
                end
            end
            L2x = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(4*(nx-4)*(ny-4)*(nz-4)); J = zeros(4*(nx-4)*(ny-4)*(nz-4)); vals = zeros(T,4*(nx-4)*(ny-4)*(nz-4));
            counter = -3;

            for i = 3:nx-2
                for j = 3:ny-2
                    for k = 3:nz-2
                        counter = counter + 4;
                        # y part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i,j+1,k);
                        indexMinus = vectorIndex(nx,ny,i,j-1,k);
                        indexPP = vectorIndex(nx,ny,i,j+2,k);
                        indexMM = vectorIndex(nx,ny,i,j-2,k);

                        if j > 2
                            II[counter] = index;
                            J[counter] = indexMM;
                            vals[counter] = 1/12/settings.dx/density[i,j-2,k];
                        end
                        if j > 1
                            II[counter+1] = index;
                            J[counter+1] = indexMinus;
                            vals[counter+1] = -8/12/settings.dx/density[i,j-1,k];
                        end
                        if j < ny
                            II[counter+2] = index;
                            J[counter+2] = indexPlus;
                            vals[counter+2] = 8/12/settings.dx/density[i,j+1,k]; 
                        end
                        if j < ny-1
                            II[counter+3] = index;
                            J[counter+3] = indexPP;
                            vals[counter+3] = -1/12/settings.dx/density[i,j+2,k]; 
                        end
                    end
                end
            end
            L2y = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(4*(nx-4)*(ny-4)*(nz-4)); J = zeros(4*(nx-4)*(ny-4)*(nz-4)); vals = zeros(T,4*(nx-4)*(ny-4)*(nz-4));
            counter = -3;

            for i = 3:nx-2
                for j = 3:ny-2
                    for k = 3:nz-2
                        counter = counter + 4;
                        # z part
                        index = vectorIndex(nx,ny,i,j,k);
                        indexPlus = vectorIndex(nx,ny,i,j,k+1);
                        indexMinus = vectorIndex(nx,ny,i,j,k-1);
                        indexPP = vectorIndex(nx,ny,i,j,k+2);
                        indexMM = vectorIndex(nx,ny,i,j,k-2);

                        if k > 2
                            II[counter] = index;
                            J[counter] = indexMM;
                            vals[counter] = 1/12/settings.dx/density[i,j,k-2];
                        end
                        if k > 1
                            II[counter+1] = index;
                            J[counter+1] = indexMinus;
                            vals[counter+1] = -8/12/settings.dx/density[i,j,k-1];
                        end
                        if k < nz
                            II[counter+2] = index;
                            J[counter+2] = indexPlus;
                            vals[counter+2] = +8/12/settings.dx/density[i,j,k+1]; 
                        end
                        if k < nz-1
                            II[counter+3] = index;
                            J[counter+3] = indexPP;
                            vals[counter+3] = -1/12/settings.dx/density[i,j,k+2]; 
                        end
                    end
                end
            end
            L2z = sparse(II,J,vals,nx*ny*nz,nx*ny*nz);

            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(T,2*(nx-2)*(ny-2)*(nz-2));
            counter = -1;
        end

        # collect boundary indices
        boundaryIdx = zeros(Int,2*nx*ny+2*ny*nz + 2*nx*nz)
        counter = 0;
        for i = 1:nx
            for k = 1:nz
                counter +=1;
                j = 1;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
                counter +=1;
                j = ny;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
            end
        end

        for i = 1:nx
            for j = 1:ny
                counter +=1;
                k = 1;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
                counter +=1;
                k = nz;
                boundaryIdx[counter] = vectorIndex(nx,ny,i,j,k)
            end
        end
        new{T}(L1x,L1y,L1z,L2x,L2y,L2z)
    end
end