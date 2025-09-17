__precompile__

using SparseArrays

include("utils.jl")

struct Stencils{T<:AbstractFloat}
    L1x::SparseMatrixCSC{T, Int64};
    L1y::SparseMatrixCSC{T, Int64};
    L2x::SparseMatrixCSC{T, Int64};
    L2y::SparseMatrixCSC{T, Int64};

    deltaox::SparseMatrixCSC{T, Int64};
    deltaoy::SparseMatrixCSC{T, Int64};
    Dox::SparseMatrixCSC{T, Int64};
    Doy::SparseMatrixCSC{T, Int64};

    function Stencils(settings::settings2D,T::DataType=Float64)
        density = settings.density;
        # setup stencil matrix
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        L1x = spzeros(nx*ny,nx*ny);
        L1y = spzeros(nx*ny,nx*ny);
        L2x = spzeros(nx*ny,nx*ny);
        L2y = spzeros(nx*ny,nx*ny);
        #these matrices are only needed for staggered grids
        deltaox = nothing
        deltaoy = nothing
        Dox = nothing
        Doy = nothing

        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3*(nx-2)*(ny-2)); J = zeros(3*(nx-2)*(ny-2)); vals = zeros(T,3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # x part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i+1,j);
                indexMinus = vectorIndex(ny,i-1,j);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dx/density[i,j]; 
                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx/density[i-1,j];
                end
                if i < nx
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dx/density[i+1,j]; 
                end
            end
        end
        L1x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(3*(nx-2)*(ny-2)); J .= zeros(3*(nx-2)*(ny-2)); vals .= zeros(T,3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # y part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i,j+1);
                indexMinus = vectorIndex(ny,i,j-1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dy/density[i,j]; 

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy/density[i,j-1];
                end
                if j < ny
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dy/density[i,j+1]; 
                end
            end
        end
        L1y = sparse(II,J,vals,nx*ny,nx*ny);

        II = zeros(2*(nx-2)*(ny-2)); J = zeros(2*(nx-2)*(ny-2)); vals = zeros(T,2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # x part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i+1,j);
                indexMinus = vectorIndex(ny,i-1,j);

                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx/density[i-1,j];
                end
                if i < nx
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dx/density[i+1,j];
                end
            end
        end
        L2x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(2*(nx-2)*(ny-2)); J .= zeros(2*(nx-2)*(ny-2)); vals = zeros(T,2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # y part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i,j+1);
                indexMinus = vectorIndex(ny,i,j-1);

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy/density[i,j-1];
                end
                if j < ny
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dy/density[i,j+1];
                end
            end
        end
        L2y = sparse(II,J,vals,nx*ny,nx*ny);
        new{T}(L1x,L1y,L2x,L2y,deltaox, deltaoy, Dox, Doy)
    end

    function Stencils(settings::settings2D,T::DataType=Float64,staggered::String="staggered")
        #staggered grid

        # setup stencil matrices
        nx = settings.NCellsX;
        ny = settings.NCellsY;
        L1x = spzeros(nx*ny,nx*ny);
        L1y = spzeros(nx*ny,nx*ny);
        L2x = spzeros(nx*ny,nx*ny);
        L2y = spzeros(nx*ny,nx*ny);
        deltaox = spzeros(nx*ny,nx*ny);
        deltaoy = spzeros(nx*ny,nx*ny);
        Dox = spzeros(nx*ny,nx*ny);
        Doy = spzeros(nx*ny,nx*ny);

        # setup index arrays and values for allocation of stencil matrices
        II = zeros(3*(nx-2)*(ny-2)); J = zeros(3*(nx-2)*(ny-2)); vals = zeros(T,3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # x part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i+1,j);
                indexMinus = vectorIndex(ny,i-1,j);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dx; 
                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx;
                end
                if i < nx
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dx; 
                end
            end
        end
        L1x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(3*(nx-2)*(ny-2)); J .= zeros(3*(nx-2)*(ny-2)); vals .= zeros(T,3*(nx-2)*(ny-2));
        counter = -2;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 3;
                # y part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i,j+1);
                indexMinus = vectorIndex(ny,i,j-1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = 2.0/2/settings.dy; 

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy;
                end
                if j < ny
                    II[counter+2] = index;
                    J[counter+2] = indexPlus;
                    vals[counter+2] = -1/2/settings.dy; 
                end
            end
        end
        L1y = sparse(II,J,vals,nx*ny,nx*ny);

        II = zeros(2*(nx-2)*(ny-2)); J = zeros(2*(nx-2)*(ny-2)); vals = zeros(T,2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # x part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i+1,j);
                indexMinus = vectorIndex(ny,i-1,j);

                if i > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dx;
                end
                if i < nx
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dx;
                end
            end
        end
        L2x = sparse(II,J,vals,nx*ny,nx*ny);

        II .= zeros(2*(nx-2)*(ny-2)); J .= zeros(2*(nx-2)*(ny-2)); vals = zeros(T,2*(nx-2)*(ny-2));
        counter = -1;

        for i = 2:nx-1
            for j = 2:ny-1
                counter = counter + 2;
                # y part
                index = vectorIndex(ny,i,j);
                indexPlus = vectorIndex(ny,i,j+1);
                indexMinus = vectorIndex(ny,i,j-1);

                if j > 1
                    II[counter] = index;
                    J[counter] = indexMinus;
                    vals[counter] = -1/2/settings.dy;
                end
                if j < ny
                    II[counter+1] = index;
                    J[counter+1] = indexPlus;
                    vals[counter+1] = 1/2/settings.dy;
                end
            end
        end
        L2y = sparse(II,J,vals,nx*ny,nx*ny);
        new{T}(L1x,L1y,L2x,L2y,deltaox, deltaoy, Dox, Doy)
    end
end