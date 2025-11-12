using LinearAlgebra
using TensorToolbox

# Define TTN struct
mutable struct TTN
    id::Int
    leaves::Vector{TTN}
    C::Array # core tensor or basis matrix
    S::Matrix # core tensor or basis matrix
    VᵀFV::Matrix
end

# Define TTNO struct
mutable struct TTNO
    FY::Vector{TTN}
end

# Constructor for TTN
function TTN(data::Int, C::Array)
    return TTN(data, TTN[], C, Matrix(undef,0,0), Matrix(undef,0,0))
end

# Constructor for TTN
function TTN(data::Int, dummy::Vector{Any}, C::Array)
    return TTN(data, TTN[], C, Matrix(undef,0,0), Matrix(undef,0,0))
end

# Constructor for TTN
function TTN(data::Int, leaves::Vector{TTN}, C::Array)
    return TTN(data, leaves, C, Matrix(undef,0,0), Matrix(undef,0,0))
end

# Function to add a leaf node to a parent node
function add_leaf!(parent::TTN, leaf::TTN)
    push!(parent.leaves, leaf)
end

function get_leaf_nodes(root::TTN)
    if isempty(root.leaves)
        return [root]
    else
        leaf_nodes = TTN[]
        for leaf in root.leaves
            append!(leaf_nodes, get_leaf_nodes(leaf))
        end
        return leaf_nodes
    end
end

function get_connecting_nodes(root::TTN)
    if isempty(root.leaves)
        return []
    else
        connecting_nodes = [root]
        for leaf in root.leaves
            append!(connecting_nodes, get_connecting_nodes(leaf))
        end
        return connecting_nodes
    end
end

# Function to print the tree structure
function print_tree(root::TTN, level::Int = 0)
    println("\t" ^ level, "Data: ", root.id)
    println("\t" ^ level, "Dim: ", size(root.C))
    for leaf in root.leaves
        print_tree(leaf, level + 1)
    end
end

# Function to copy a subtree
function copy_subtree(node::TTN)
    new_node = TTN(node.id, Vector{TTN}[], Base.deepcopy(node.C), Base.deepcopy(node.S), Base.deepcopy(node.VᵀFV))
    for leaf in node.leaves
        new_leaf = copy_subtree(leaf)
        add_leaf!(new_node, new_leaf)
    end
    return new_node
end

# computes the inner product of two TTNs
function inner(Y::TTN, Z::TTN)
    if length(Y.leaves) != 0 && length(Z.leaves) != 0 # node is connecting node
        Uτᵢ = Matrix{Float64}[];
        for (leafY, leafZ) in zip(Y.leaves, Z.leaves)
            push!(Uτᵢ, inner(leafY, leafZ)')
        end
        return tenmat(ttm(Y.C, Uτᵢ, collect(2:length(Y.leaves)+1)), 1)*tenmat(Z.C, 1)'
    elseif length(Y.leaves) == 0 && length(Z.leaves) == 0 # node is leaf
        return Y.C' * Z.C
    else
        print_tree(Y)
        print_tree(Z)
        error("Error in inner: Y and Z do not have same length and structure: ", size(Y.C), " vs. ", size(Z.C))
    end
end

# multiplies the TTN Y with bases Xᵢ and evaluates the resulting tensor. Compared to matrix view: input Y = USVᵀ, X = [U,V]. Returns UᵀUSVᵀV (= S)
function project_and_eval(Y::TTN, X::Vector{TTN})
    A = Matrix{Float64}[]
    for (Yᵢ, Xᵢ) in zip(Y.leaves, X)
        push!(A, inner(Xᵢ, Yᵢ))
    end
    return ttm(Y.C, A, collect(1:length(Y.leaves)) .+ 1)
end

# Y is current TTN
# FY is F evaluated at Y
# the test tensor V consists of the corresponding subtree τ in Y where the subtrees node is replaced by Q
function prolong_and_retract(FY::Vector{TTN}, Y::TTN, i::Int)
    noti = collect(1:length(Y.leaves)); deleteat!(noti, i);
    FᵢY = TTN[]
    for FYₖ in FY
        FᵢYₖ = copy_subtree(FYₖ.leaves[i])
        UᵀAUτⱼ = Matrix{Float64}[];
        for j in noti # compute all projections of τ's leaves (except for leaf i) 
            push!(UᵀAUτⱼ, inner(Y.leaves[j], FYₖ.leaves[j]))
        end
        #¬ = ¬ .+ 1 # leaf j is at position j+1 in Q, since we have direction 0 at 1. Hence all indices must be increased by 1.
        FᵢYₖ.VᵀFV = tenmat(ttm(FYₖ.C, UᵀAUτⱼ, noti .+ 1), i+1)*tenmat(Y.C, i+1)' # compute final projection with respect to Q.  Y.C stores Q.
        push!(FᵢY, FᵢYₖ)
    end
    return FᵢY
end

# Y is current TTN
# FY is F evaluated at Y
# the test tensor V consists of the corresponding subtree τ in Y where the subtrees node is replaced by Q
# "full" means that Qᵀ, Sᵀ still need to be computed
function prolong_and_retract_full(FY::Vector{TTN}, Y::TTN, i::Int)
    noti = collect(1:length(Y.leaves)); deleteat!(noti, i);

    # perform QR with QᵀSᵀ = Matᵢ(Y.C)ᵀ so that the test funtions will be orthonormal.
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(ttm(Y.C, Y.S, 1), i+1)', mode="reduced");
    Qᵢ = matten(Matrix(Qᵀ'), i+1, [rᵢ for rᵢ in size(Y.C)])
    Y.leaves[i].S = Matrix(Sᵀ') # move the information (S) onto the i-th subtree. This information will be needed later in the RK steps when computing K 

    FᵢY = TTN[]
    for FYₖ in FY
        #QFᵀ, SFᵀ = np.linalg.qr(tenmat(ttm(FYₖ.C, FYₖ.VᵀFV * FYₖ.S, 1), i+1)', mode="reduced");
        QFᵀ, SFᵀ = np.linalg.qr(tenmat(ttm(FYₖ.C, FYₖ.VᵀFV * FYₖ.S , 1), i+1)', mode="reduced"); # this needs to be checked! The transpose in FYₖ.VᵀFV' is new!
        rC = collect(size(FYₖ.C))
        rC[1] = size(Y.C, 1)
        QFᵢ = matten(Matrix(QFᵀ'), i+1, [rᵢ for rᵢ in rC])
        FYₖ.leaves[i].S = Matrix(SFᵀ')
        UᵀAUτⱼ = Matrix{Float64}[];
        for j in noti # compute all projections of τ's leaves (except for leaf i) 
            push!(UᵀAUτⱼ, inner(Y.leaves[j], FYₖ.leaves[j])) # this does not transpose flux matrices
        end

        FYₖ.leaves[i].VᵀFV = tenmat(Qᵢ, i+1)*tenmat(ttm(QFᵢ, UᵀAUτⱼ, noti .+ 1), i+1)' # compute final projection with respect to Q.  
        # The above step transposes the flux matrix again, so we recover the original matrix. The V from the solution is on the front (test functions), the V from the TTNO is on the back (the V of the solution to which F is applied).
        push!(FᵢY, FYₖ.leaves[i])
    end
    return Sᵀ, FᵢY
end

# Y is current TTN
# FY is F evaluated at Y
# the test tensor V consists of the corresponding subtree τ in Y where the subtrees node is replaced by Q
# "full" means that Qᵀ, Sᵀ still need to be computed
function prolong_and_retract_full_bu(FY::Vector{TTN}, Y::TTN, i::Int)
    noti = collect(1:length(Y.leaves)); deleteat!(noti, i);

    # perform QR with QᵀSᵀ = Matᵢ(Y.C)ᵀ so that the test funtions will be orthonormal.
    Qᵀ, Sᵀ = np.linalg.qr(tenmat(ttm(Y.C, Y.S, 1), i+1)', mode="reduced");
    Qᵢ = matten(Matrix(Qᵀ'), i+1, [rᵢ for rᵢ in size(Y.C)])
    Y.leaves[i].S = Matrix(Sᵀ') # move the information (S) onto the i-th subtree. This information will be needed later in the RK steps when computing K 

    FᵢY = TTN[]
    for FYₖ in FY
        #println(size(FYₖ.C))
        #println(size(FYₖ.VᵀFV))
        #println(size(FYₖ.S))
        #println("check symmetry: ", norm(FYₖ.VᵀFV - FYₖ.VᵀFV'))
        #QFᵀ, SFᵀ = np.linalg.qr(tenmat(ttm(FYₖ.C, FYₖ.VᵀFV * FYₖ.S, 1), i+1)', mode="reduced");
        QFᵀ, SFᵀ = np.linalg.qr(tenmat(ttm(FYₖ.C, FYₖ.VᵀFV' * FYₖ.S, 1), i+1)', mode="reduced"); # this needs to be checked! The transpose in FYₖ.VᵀFV' is new!
        #println(size(Matrix(QFᵀ')))
        #println(size(FYₖ.C))
        #println(size(Y.C))
        rC = collect(size(FYₖ.C))
        rC[1] = size(Y.C, 1)
        QFᵢ = matten(Matrix(QFᵀ'), i+1, [rᵢ for rᵢ in rC])
        FYₖ.leaves[i].S = Matrix(SFᵀ')
        UᵀAUτⱼ = Matrix{Float64}[];
        for j in noti # compute all projections of τ's leaves (except for leaf i) 
            push!(UᵀAUτⱼ, inner(FYₖ.leaves[j], Y.leaves[j])')
        end
        FYₖ.leaves[i].VᵀFV = tenmat(ttm(QFᵢ, UᵀAUτⱼ, noti .+ 1), i+1)*tenmat(Qᵢ, i+1)' # compute final projection with respect to Q.  Y.C stores Q.
        push!(FᵢY, FYₖ.leaves[i])
    end
    return Sᵀ, FᵢY
end


# the test tensor V consists of the corresponding subtree τ in Y where the subtrees node is replaced by Q
function prolong_and_retract_full(𝑄::TTN, Y::TTN, i::Int)
    noti = collect(1:length(Y.leaves)); deleteat!(noti, i);
    𝑄ᵢ = copy_subtree(𝑄)
    Y = copy_subtree(Y)

    Qᵀ, Sᵀ = np.linalg.qr(tenmat(ttm(Y.C, Y.S, 1), i+1)', mode="reduced");
    Qᵢ = matten(Matrix(Qᵀ'), i+1, [rᵢ for rᵢ in size(Y.C)])

    Y.leaves[i].S = Matrix(Sᵀ')
    UᵀAUτⱼ = Matrix{Float64}[];
    for j in noti # compute all projections of τ's leaves (except for leaf i) 
        push!(UᵀAUτⱼ, inner(𝑄ᵢ.leaves[j], Y.leaves[j]))
    end
    𝑄ᵢ.leaves[i].VᵀFV = tenmat(ttm(Qᵢ, UᵀAUτⱼ, noti .+ 1), i+1)*tenmat(Qᵢ, i+1)' # compute final projection with respect to Q.  Y.C stores Q.
    return Sᵀ, 𝑄ᵢ.leaves[i]
end

function get_rank(obj::TTN, r=[], isRoot=true)
    if length(obj.leaves) != 0 # obj is connecting node
        push!(r, size(obj.C, 1))
        for leaf in obj.leaves
            get_rank(leaf, r, false)
        end
    else
        push!(r, size(obj.C, 2))
    end
    if isRoot
        return r
    end
end

function eval(obj::TTN, idx::Vector{Any}, Y=nothing)
    if Y === nothing # node is root
        Y = obj
        Uτᵢ = Matrix{Float64}[];
        for leaf in Y.leaves
            push!(Uτᵢ, eval(obj, idx, leaf))
        end
        return tenmat(ttm(Y.C, Uτᵢ, collect(2:length(Y.leaves)+1)), 1)
    elseif length(Y.leaves) != 0 # Y is connecting Y
        Uτᵢ = Matrix{Float64}[];
        for leaf in Y.leaves
            push!(Uτᵢ, eval(obj, idx, leaf))
        end
        return tenmat(ttm(Y.C, Uτᵢ, collect(2:length(Y.leaves)+1)), 1)'
    else # Y is leaf
        i = Y.id
        if typeof(idx[i]) == Colon
            len = size(Y.C, 1)
        else
            len = length(idx[i])
        end
        return reshape(Y.C[idx[i],:] , len, size(Y.C, 2))
    end
end

function eval(obj::TTN, idx::Vector{Int}, Y=nothing)
    if Y === nothing # node is root
        Y = obj
        Uτᵢ = Matrix{Float64}[];
        for leaf in Y.leaves
            push!(Uτᵢ, eval(obj, idx, leaf))
        end
        return tenmat(ttm(Y.C, Uτᵢ, collect(2:length(Y.leaves)+1)), 1)[1]
    elseif length(Y.leaves) != 0 # Y is connecting Y
        Uτᵢ = Matrix{Float64}[];
        for leaf in Y.leaves
            push!(Uτᵢ, eval(obj, idx, leaf))
        end
        return tenmat(ttm(Y.C, Uτᵢ, collect(2:length(Y.leaves)+1)), 1)'
    else # Y is leaf
        i = Y.id
        if typeof(idx[i]) == Colon
            len = size(Y.C, 1)
        else
            len = length(idx[i])
        end
        return reshape(Y.C[idx[i],:] , len, size(Y.C, 2))
    end
end

# shifts the orthonormal matrix Q up and information S down to leaf i
function shiftOrthogonal!(obj::TTN, Qᵢ::Array, Sᵀ::Matrix, i)
    obj.C  .= Qᵢ
    obj.leaves[i].S = Matrix(Sᵀ')
    return Sᵀ
end

# shifts the orthonormal matrix Q up and information S down to leaf i
function shiftOrthogonal!(obj::Vector{TTN}, Qᵢ::Array, Sᵀ::Matrix, i)
    for FYᵢ in obj
        FYᵢ.C  .= Qᵢ
        FYᵢ.leaves[i].S = Matrix(Sᵀ')
    end
end

# copies TTNO
function copy(obj::Vector{TTN})
    FY = TTN[]
    for FYᵢ in obj
        push!(FY, copy_subtree(FYᵢ))
    end
    return FY
end

# Example usage:

# Create original tree
function generateTestTree(r=3, n=10)
    root = TTN(7, 7*ones(1,r,r))
    leaf1 = TTN(6, ones(r,r,r))
    leaf2 = TTN(5, 2*ones(r,r,r))
    subleaf1 = TTN(1, 3*ones(n,r))
    subleaf2 = TTN(2, 4*ones(n,r))
    subleaf3 = TTN(3, 5*ones(n,r))
    subleaf4 = TTN(4, 6*ones(n,r))
    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateOrthTestTree(r=2, n=15)
    x = range(-1, 1, n)
    input = zeros(n, r)
    σ² = 0.01
    input[:, 1] .= exp.(-x.^2 / σ²)
    Q, R = np.linalg.qr(input, mode="reduced");
    subleaf1 = TTN(1, Q)
    subleaf2 = TTN(2, Q)
    subleaf3 = TTN(3, Q)
    subleaf4 = TTN(4, Q)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for i = 2:3
        input = R * tenmat(input, i)
        input = matten(input, i, [r, r, r])
    end
    Q, R = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q = matten(Q', 1, [r, r, r])
    leaf1 = TTN(6, Q)
    leaf2 = TTN(5, Q)

    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for i = 2:3
        input = R * tenmat(input, i)
        input = matten(input, i, [1, r, r])
    end
    root = TTN(7, input)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateRadTree(s::Settings)
    Q = []
    R = []
    QUQ = []
    RUQ = []
    n = s.NCells
    nΩ = s.nPN
    nξ = s.Nxi
    nη = s.Neta
    r = s.r
    input = zeros(n, r)
    input[:, 1] .= IC(s.xMid)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf1 = TTN(1, Qᵢ) # space

    input = zeros(nΩ, r)
    input[1, 1] = 1
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf2 = TTN(2, Qᵢ) # angle (moments)

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf3 = TTN(3, Qᵢ) # ξ

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf4 = TTN(4, Qᵢ) # η

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₁ = matten(Q₁', 1, [r, r, r])
    leaf1 = TTN(3, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₂ = matten(Q₂', 1, [r, r, r])
    leaf2 = TTN(2, Q₂)

    Rroot = [R₁, R₂]
    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(Rroot)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    root = TTN(1, input)
    root.S = ones(Float64,1,1)
    root.VᵀFV = ones(Float64,1,1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateRadTree2D(s::Settings)
    Q = []
    R = []
    QUQ = []
    RUQ = []
    n = s.NCells^2
    nΩ = GlobalIndex( s.nPN, s.nPN ) + 1
    nξ = s.Nxi
    nη = s.Neta
    r = s.r
    input = zeros(n, r)
    
    if s.problem == "Lattice"
        input[:, 1] .= 1e-5
    else
        input[:, 1] .= vec(IC(s, s.xMid, s.xMid))
    end
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf1 = TTN(1, Qᵢ) # space

    input = zeros(nΩ, r)
    input[1, 1] = 1
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf2 = TTN(2, Qᵢ) # angle (moments)

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf3 = TTN(3, Qᵢ) # ξ

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf4 = TTN(4, Qᵢ) # η

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₁ = matten(Q₁', 1, [r, r, r])
    leaf1 = TTN(3, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₂ = matten(Q₂', 1, [r, r, r])
    leaf2 = TTN(2, Q₂)

    Rroot = [R₁, R₂]
    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(Rroot)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    root = TTN(1, input)
    root.S = ones(Float64,1,1)
    root.VᵀFV = ones(Float64,1,1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateRadTree3D(s::Settings)
    Q = []
    R = []
    QUQ = []
    RUQ = []
    n = s.NCells^3
    nΩ = GlobalIndex( s.nPN, s.nPN ) + 1
    nξ = s.Nxi
    nη = s.Neta
    r = s.r
    input = zeros(n, r)
    input[:, 1] .= vec(IC(s, s.xMid, s.xMid, s.xMid))
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf1 = TTN(1, Qᵢ) # space

    input = zeros(nΩ, r)
    input[1, 1] = 1
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf2 = TTN(2, Qᵢ) # angle (moments)

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf3 = TTN(3, Qᵢ) # ξ

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf4 = TTN(4, Qᵢ) # η

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₁ = matten(Q₁', 1, [r, r, r])
    leaf1 = TTN(3, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₂ = matten(Q₂', 1, [r, r, r])
    leaf2 = TTN(2, Q₂)

    Rroot = [R₁, R₂]
    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(Rroot)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    root = TTN(1, input)
    root.S = ones(Float64,1,1)
    root.VᵀFV = ones(Float64,1,1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateSource(s::Settings, source)
    Q = []
    R = []
    QUQ = []
    RUQ = []
    n = s.NCells^2
    nΩ = GlobalIndex( s.nPN, s.nPN ) + 1
    nξ = s.Nxi
    nη = s.Neta
    r = 1
    input = zeros(n, r)
    input[:, 1] .= source
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf1 = TTN(1, Qᵢ) # space

    input = zeros(nΩ, r)
    input[1, 1] = 1
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf2 = TTN(2, Qᵢ) # angle (moments)

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf3 = TTN(3, Qᵢ) # ξ

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf4 = TTN(4, Qᵢ) # η

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₁ = matten(Q₁', 1, [r, r, r])
    leaf1 = TTN(3, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₂ = matten(Q₂', 1, [r, r, r])
    leaf2 = TTN(2, Q₂)

    Rroot = [R₁, R₂]
    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(Rroot)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    root = TTN(1, input)
    root.S = ones(Float64,1,1)
    root.VᵀFV = ones(Float64,1,1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateIsingTree(s::Settings)
    Q = []
    R = []
    QUQ = []
    RUQ = []
    n = s.NCells
    nΩ = s.nPN
    nξ = s.Nxi
    nη = s.Neta
    r = s.r
    input = Diagonal([1.0, 1.0])
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf1 = TTN(1, Qᵢ) # particle 1

    input = Diagonal([1.0, 1.0])
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf2 = TTN(2, Qᵢ) # particle 2

    input = Diagonal([1.0, 1.0])
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf3 = TTN(3, Qᵢ) # particle 3

    input = Diagonal([1.0, 1.0])
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subleaf4 = TTN(4, Qᵢ) # particle 4

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₁ = matten(Q₁', 1, [r, r, r])
    leaf1 = TTN(3, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₂ = matten(Q₂', 1, [r, r, r])
    leaf2 = TTN(2, Q₂)

    Rroot = [R₁, R₂]
    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(Rroot)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    root = TTN(1, input)
    root.S = ones(Float64,1,1)
    root.VᵀFV = ones(Float64,1,1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    return root
end

# Create original tree
function generateRadTree8D(s::Settings)
    Q = []
    R = []
    QUQ = []
    RUQ = []
    QUQ₁ = []
    RUQ₁ = []
    QUQ₂ = []
    RUQ₂ = []
    QUQ2 = []
    RUQ2 = []
    QUQ3 = []
    RUQ3 = []
    n = s.NCells
    nΩ = s.nPN
    nξ = s.Nxi
    nη = s.Neta
    r = s.r
    input = zeros(n, r)
    input[:, 1] .= IC(s.xMid)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subsubleaf1 = TTN(1, Qᵢ) # space

    input = zeros(nΩ, r)
    input[1, 1] = 1
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subsubleaf2 = TTN(2, Qᵢ) # angle (moments)

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subsubleaf3 = TTN(3, Qᵢ) # ξ

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ, Qᵢ); push!(RUQ, Rᵢ)
    subsubleaf4 = TTN(4, Qᵢ) # η

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ₁, Qᵢ); push!(RUQ₁, Rᵢ)
    subsubleaf5 = TTN(5, Qᵢ) # ξ₁

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ₁, Qᵢ); push!(RUQ₁, Rᵢ)
    subsubleaf6 = TTN(6, Qᵢ) # η₁

    input = ones(nξ, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ₂, Qᵢ); push!(RUQ₂, Rᵢ)
    subsubleaf7 = TTN(7, Qᵢ) # ξ₂

    input = ones(nη, r)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(QUQ₂, Qᵢ); push!(RUQ₂, Rᵢ)
    subsubleaf8 = TTN(8, Qᵢ) # η₂

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    push!(QUQ2, Q₁); push!(RUQ2, R₁)
    Q₁ = matten(Q₁', 1, [r, r, r])
    subleaf1 = TTN(7, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    push!(QUQ2, Q₂); push!(RUQ2, R₂)
    Q₂ = matten(Q₂', 1, [r, r, r])
    subleaf2 = TTN(6, Q₂)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ₁)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    push!(QUQ3, Q₁); push!(RUQ3, R₁)
    Q₁ = matten(Q₁', 1, [r, r, r])
    subleaf3 = TTN(5, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ₂)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    push!(QUQ3, Q₂); push!(RUQ3, R₂)
    Q₂ = matten(Q₂', 1, [r, r, r])
    subleaf4 = TTN(4, Q₂)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ2)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₁, R₁ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₁ = matten(Q₁', 1, [r, r, r])
    leaf1 = TTN(3, Q₁)

    input = zeros(r, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(RUQ3)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [r, r, r])
    end
    Q₂, R₂ = np.linalg.qr(tenmat(input, 1)', mode="reduced");
    Q₂ = matten(Q₂', 1, [r, r, r])
    leaf2 = TTN(2, Q₂)

    Rroot = [R₁, R₂]
    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(Rroot)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    root = TTN(1, input)
    root.S = ones(Float64,1,1)
    root.VᵀFV = ones(Float64,1,1)

    add_leaf!(root, leaf1)
    add_leaf!(root, leaf2)
    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    add_leaf!(leaf2, subleaf3)
    add_leaf!(leaf2, subleaf4)
    add_leaf!(subleaf1, subsubleaf1)
    add_leaf!(subleaf1, subsubleaf2)
    add_leaf!(subleaf2, subsubleaf3)
    add_leaf!(subleaf2, subsubleaf4)
    add_leaf!(subleaf3, subsubleaf5)
    add_leaf!(subleaf3, subsubleaf6)
    add_leaf!(subleaf4, subsubleaf7)
    add_leaf!(subleaf4, subsubleaf8)
    return root
end

# Create original tree
function generateSmallRadTree(s::Settings)
    Q = []
    R = []
    n = s.NCells
    nΩ = s.nPN
    r = s.r
    input = zeros(n, r)
    input[:, 1] .= IC(s.xMid)
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf1 = TTN(1, Qᵢ) # space

    input = zeros(nΩ, r)
    input[1, 1] = 1
    input[2, 1] = 1
    Qᵢ, Rᵢ = np.linalg.qr(input, mode="reduced");
    push!(Q, Qᵢ); push!(R, Rᵢ)
    subleaf2 = TTN(2, Qᵢ) # angle (moments)

    input = zeros(1, r, r)
    input[1, 1, 1] = 1.0
    for (i, Rᵢ) in enumerate(R)
        input = Rᵢ * tenmat(input, i+1)
        input = matten(input, i+1, [1, r, r])
    end
    leaf1 = TTN(1, input)
    leaf1.S = ones(Float64,1,1)
    leaf1.VᵀFV = ones(Float64,1,1)

    add_leaf!(leaf1, subleaf1)
    add_leaf!(leaf1, subleaf2)
    return leaf1
end
