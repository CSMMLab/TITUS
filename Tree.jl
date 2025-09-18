using AbstractTrees

struct IDTreeNode
    id::Int
    parent::Union{Nothing,IDTreeNode};
    leaves::Vector{IDTreeNode}

    IDTreeNode(id::Integer, parent=nothing, leaves::Vector{IDTreeNode}=IDTreeNode[]) = new(id, parent, leaves)
end

AbstractTrees.nodevalue(n::IDTreeNode) = n.id

AbstractTrees.parent(n::IDTreeNode) = n.parent

AbstractTrees.children(node::IDTreeNode) = node.leaves
AbstractTrees.printnode(io::IO, node::IDTreeNode) = print(io, "#", node.id)

function addleaf!(parent::IDTreeNode, data::Int)
    tmp = IDTreeNode(data, parent);
    push!(parent.leaves, tmp)
    return tmp;
end

"""
    IDTree

Basic tree type used for testing.

Each node has a unique ID, making them easy to reference. Node leaves are ordered.

Node type only implements `leaves`, so serves to test default implementations of most functions.
"""
struct IDTree
    nodes::Dict{Int,IDTreeNode}
    root::IDTreeNode
end

"""
    IDTree(x)

Create from nested `id => leaves` pairs. Leaf nodes may be represented by ID only.
"""
function IDTree(x)
    root = _make_idtreenode(x)
    nodes = Dict{Int, IDTreeNode}()

    for node in PreOrderDFS(root)
        haskey(nodes, node.id) && error("Duplicate node ID $(node.id)")
        nodes[node.id] = node
    end

    return IDTree(nodes, root)
end

function idnode_example()
    root = IDTreeNode(0)
    i1 = addchild!(root, 1)
    i2 = addchild!(root, 2)
    j1 = addchild!(i1, 3)

    n₀
end