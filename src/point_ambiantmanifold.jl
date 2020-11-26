
#
### Point / vector wrappers
#
@enum ReprType ambiant_repr manifold_repr
mutable struct Point{Tambiantpoint, Tmanifoldpoint}
    amb_repr::Tambiantpoint
    man_repr::Tmanifoldpoint
    M::Manifold
    repr::ReprType
end

function copy(p::Point)
    return Point(deepcopy(p.amb_repr), deepcopy(p.man_repr), deepcopy(p.M), p.repr)
end

embed(x) = x
function embed(x::Point)
    return x.repr == ambiant_repr ? x.amb_repr : embed(x.M, x.man_repr)
end

get_repr(x) = x
function get_repr(x::Point)
    return x.repr == ambiant_repr ? x.amb_repr : x.man_repr
end



function convert_point_repr!(ostate, optimizer, pb)
    convert_point_repr!(ostate.x, ostate.selected_update)
    return
end

