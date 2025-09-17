__precompile__

function sub2ind(s,row,col)
    return LinearIndices(s)[CartesianIndex.(row,col)]
end

mutable struct PNSystem{T<:AbstractFloat}
    # symmetric flux matrices
    Ax::CuSparseMatrixCSC{T, Int32};
    Ay::CuSparseMatrixCSC{T, Int32};
    Az::CuSparseMatrixCSC{T, Int32};

    # Solver settings
    settings::Settings;

    # total number of moments
    nTotalEntries::Int
    # degree of expansion
    N::Int

    M::SparseMatrixCSC{ComplexF64, Int};

    T::DataType;

    # Roe matrices
    AbsAx::CuArray{T, 2};
    AbsAy::CuArray{T, 2};
    AbsAz::CuArray{T, 2};

    # decomposed flux matrices
    Vx::CuArray{T, 2}
    Λx⁺::Diagonal{T, CuArray{T, 1}}
    Λx⁻::Diagonal{T, CuArray{T, 1}}

    Vy::CuArray{T, 2}
    Λy⁺::Diagonal{T, CuArray{T, 1}}
    Λy⁻::Diagonal{T, CuArray{T, 1}}

    Vz::CuArray{T, 2}
    Λz⁺::Diagonal{T, CuArray{T, 1}}
    Λz⁻::Diagonal{T, CuArray{T, 1}}

    # constructor
    function PNSystem(settings::Settings,T::DataType=Float64)
        N = settings.nPN;
        nTotalEntries = GlobalIndex( N, N ) + 1;    # total number of entries for sytem matrix

        IndI = []; IndJ = []; val = [];
        # Assemble transformation matrix
        for m = 2:N+1
            for l = 1:m-1
                r = (m.-1)^2 .+2*l;
                IndItmp = Int.([r.-1,r,r-1,r]);
                IndJtmp = Int.([(m-1)^2+l,(m-1)^2+l,m^2+1-l,m^2+1-l]);
                valTmp = [1,-1im,(-1).^(m+l),(-1).^(m+l)*1im]./sqrt(2);
                IndI = [IndI;IndItmp];
                IndJ = [IndJ;IndJtmp];
                val = [val;valTmp]
            end
        end
        for m = 1:1:N+1
            IndI = [IndI;m^2];
            IndJ = [IndJ;(m-1)^2+m];
            val = [val;1]
        end
        M = sparse(Int.(IndI),Int.(IndJ),val,nTotalEntries,nTotalEntries);

        # allocate dummy matrices. Correct matrices are reallocated in respective functions
        Ax = CuSparseMatrixCSC(sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries)));
        Ay = CuSparseMatrixCSC(sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries)));
        Az = CuSparseMatrixCSC(sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries)));

        #AbsAx = CuSparseMatrixCSC(sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries)));
        #AbsAy = CuSparseMatrixCSC(sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries)));
        #AbsAz = CuSparseMatrixCSC(sparse([Int32(1)],[Int32(1)],[T(1.0)],Int32(nTotalEntries),Int32(nTotalEntries)));

        new{T}(Ax,Ay,Az,settings,nTotalEntries,N,M,T);
    end
end

function AParam( l::Int, k::Int )
    return sqrt( ( ( l - k + 1 ) * ( l + k + 1 ) ) / ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) );
end

function BParam( l::Int, k::Int ) 
    return sqrt( ( ( l - k ) * ( l + k ) ) / ( ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) ) );
end

function CParam( l::Int, k::Int ) 
    return sqrt( ( ( l + k + 1 ) * ( l + k + 2 ) ) / ( ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) ) );
end

function DParam( l::Int, k::Int ) 
    return sqrt( ( ( l - k ) * ( l - k - 1 ) ) / ( ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) ) );
end

function EParam( l::Int, k::Int ) 
    return sqrt( ( ( l - k + 1 ) * ( l - k + 2 ) ) / ( ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) ) );
end

function FParam( l::Int, k::Int ) 
    return sqrt( ( ( l + k ) * ( l + k - 1 ) ) / ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) );
end

function CTilde( l::Int, k::Int ) 
    if k < 0  return 0.0; end
    if k == 0 
        return sqrt( 2 ) * CParam( l, k );
    else
        return CParam( l, k );
    end
end

function DTilde( l::Int, k::Int ) 
    if k < 0  return 0.0; end
    if k == 0 
        return sqrt( 2 ) * DParam( l, k );
    else
        return DParam( l, k );
    end
end

function ETilde( l::Int, k::Int ) 
    if k == 1 
        return sqrt( 2 ) * EParam( l, k );
    else
        return EParam( l, k );
    end
end

function FTilde( l::Int, k::Int ) 
    if k == 1
        return sqrt( 2 ) * FParam( l, k );
    else
        return FParam( l, k );
    end
end

function Sgn( k::Int ) 
    if k >= 0 
        return 1;
    else
        return -1;
    end
end

function GlobalIndex( l::Int, k::Int ) 
    numIndicesPrevLevel  = l * l;    # number of previous indices untill level l-1
    prevIndicesThisLevel = k + l;    # number of previous indices in current level
    return numIndicesPrevLevel + prevIndicesThisLevel;
end

function kPlus( k::Int )  return k + Sgn( k ); end

function kMinus( k::Int )  return k - Sgn( k ); end

function unsigned(x::Float64)
    return Int(floor(x))
end

function int(x::Float64)
    return Int(floor(x))
end

function SetupSystemMatrices(obj::PNSystem)
    nTotalEntries = obj.nTotalEntries;    # total number of entries for sytem matrix
    N = obj.N

    Ax = zeros(nTotalEntries,nTotalEntries)
    Ay = zeros(nTotalEntries,nTotalEntries)
    Az = zeros(nTotalEntries,nTotalEntries)

    # loop over columns of A
    for l = 0:N
        for k=-l:l
            i = GlobalIndex( l, k ) ;

            # flux matrix in direction x
            if k != -1
                j = GlobalIndex( l - 1, kMinus( k ) );
                if j >= 0 && j < nTotalEntries Ax[i+1,j+1] = 0.5 * CTilde( l - 1, abs( k ) - 1 ); end

                j = GlobalIndex( l + 1, kMinus( k ) );
                if j >= 0 && j < nTotalEntries Ax[i+1,j+1] = -0.5 * DTilde( l + 1, abs( k ) - 1 ); end
            end

            j = GlobalIndex( l - 1, kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ax[i+1,j+1] = -0.5 * ETilde( l - 1, abs( k ) + 1 ); end

            j = GlobalIndex( l + 1, kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ax[i+1,j+1] = 0.5 * FTilde( l + 1, abs( k ) + 1 ); end

            # flux matrix in direction y
            if  k != 1
                j = GlobalIndex( l - 1, -kMinus( k ) );
                if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = -0.5 * Sgn( k ) * CTilde( l - 1, abs( k ) - 1 ); end

                j = GlobalIndex( l + 1, -kMinus( k ) );
                if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = 0.5 * Sgn( k ) * DTilde( l + 1, abs( k ) - 1 ); end
            end

            j = GlobalIndex( l - 1, -kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = -0.5 * Sgn( k ) * ETilde( l - 1, abs( k ) + 1 ); end

            j = GlobalIndex( l + 1, -kPlus( k ) );
            if  j >= 0 && j < nTotalEntries  Ay[i+1,j+1] = 0.5 * Sgn( k ) * FTilde( l + 1, abs( k ) + 1 ); end

            # flux matrix in direction z
            j = GlobalIndex( l - 1, k );
            if  j >= 0 && j < nTotalEntries  Az[i+1,j+1] = AParam( l - 1, k ); end

            j = GlobalIndex( l + 1, k );
            if  j >= 0 && j < nTotalEntries  Az[i+1,j+1] = BParam( l + 1, k ); end
        end
    end
    return Ax,Ay,Az
end

function SetupSystemMatricesSparse(obj::PNSystem)
    nTotalEntries = obj.nTotalEntries
    N = obj.N

    #pre-allocate
    Ix, Jx, valsx = Int[], Int[], Float64[]
    Iy, Jy, valsy = Int[], Int[], Float64[]
    Iz, Jz, valsz = Int[], Int[], Float64[]

    # Loop over columns of A
    for l = 0:N
        for k = -l:l
            i = GlobalIndex(l, k)

            # Flux matrix in direction x
            if k != -1
                j = GlobalIndex(l - 1, kMinus(k))
                if 0 ≤ j < nTotalEntries
                    push!(Ix, i+1); push!(Jx, j+1); push!(valsx, 0.5 * CTilde(l - 1, abs(k) - 1))
                end

                j = GlobalIndex(l + 1, kMinus(k))
                if 0 ≤ j < nTotalEntries
                    push!(Ix, i+1); push!(Jx, j+1); push!(valsx, -0.5 * DTilde(l + 1, abs(k) - 1))
                end
            end

            j = GlobalIndex(l - 1, kPlus(k))
            if 0 ≤ j < nTotalEntries
                push!(Ix, i+1); push!(Jx, j+1); push!(valsx, -0.5 * ETilde(l - 1, abs(k) + 1))
            end

            j = GlobalIndex(l + 1, kPlus(k))
            if 0 ≤ j < nTotalEntries
                push!(Ix, i+1); push!(Jx, j+1); push!(valsx, 0.5 * FTilde(l + 1, abs(k) + 1))
            end

            # Flux matrix in direction y
            if k != 1
                j = GlobalIndex(l - 1, -kMinus(k))
                if 0 ≤ j < nTotalEntries
                    push!(Iy, i+1); push!(Jy, j+1); push!(valsy, -0.5 * Sgn(k) * CTilde(l - 1, abs(k) - 1))
                end

                j = GlobalIndex(l + 1, -kMinus(k))
                if 0 ≤ j < nTotalEntries
                    push!(Iy, i+1); push!(Jy, j+1); push!(valsy, 0.5 * Sgn(k) * DTilde(l + 1, abs(k) - 1))
                end
            end

            j = GlobalIndex(l - 1, -kPlus(k))
            if 0 ≤ j < nTotalEntries
                push!(Iy, i+1); push!(Jy, j+1); push!(valsy, -0.5 * Sgn(k) * ETilde(l - 1, abs(k) + 1))
            end

            j = GlobalIndex(l + 1, -kPlus(k))
            if 0 ≤ j < nTotalEntries
                push!(Iy, i+1); push!(Jy, j+1); push!(valsy, 0.5 * Sgn(k) * FTilde(l + 1, abs(k) + 1))
            end

            # Flux matrix in direction z
            j = GlobalIndex(l - 1, k)
            if 0 ≤ j < nTotalEntries
                push!(Iz, i+1); push!(Jz, j+1); push!(valsz, AParam(l - 1, k))
            end

            j = GlobalIndex(l + 1, k)
            if 0 ≤ j < nTotalEntries
                push!(Iz, i+1); push!(Jz, j+1); push!(valsz, BParam(l + 1, k))
            end
        end
    end

    obj.Ax = CuSparseMatrixCSC(sparse(Ix, Jx, obj.T.(valsx), nTotalEntries, nTotalEntries))
    obj.Ay = CuSparseMatrixCSC(sparse(Iy, Jy, obj.T.(valsy), nTotalEntries, nTotalEntries))
    obj.Az = CuSparseMatrixCSC(sparse(Iz, Jz, obj.T.(valsz), nTotalEntries, nTotalEntries))
end


function SetupRoeMatrices(obj::PNSystem)
    T = obj.T
    eig = LinearAlgebra.eigen(Matrix(obj.Ax))
    S = eig.values
    V = eig.vectors
    AbsAx = V*abs.(Diagonal(S))*inv(V)
    #idx = findall(abs.(AbsAx) .> 1e-3)
    #Ix = first.(Tuple.(idx)); Jx = last.(Tuple.(idx)); vals = AbsAx[idx];
    #obj.AbsAx = CuSparseMatrixCSC(sparse(Iz,Jz,T.(valsy),pn.nTotalEntries,pn.nTotalEntries));
    obj.AbsAx = CuArray(AbsAx);

    eig = LinearAlgebra.eigen(Matrix(obj.Ay))
    S = eig.values
    V = eig.vectors
    AbsAy = V*abs.(diagm(S))*inv(V)
    obj.AbsAy = CuArray(AbsAy);
    
    eig = LinearAlgebra.eigen(Matrix(obj.Az))
    S = eig.values
    V = eig.vectors
    AbsAz = V*abs.(diagm(S))*inv(V)
    obj.AbsAz = CuArray(AbsAz);
end
