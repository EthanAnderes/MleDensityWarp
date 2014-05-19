module GaussKernel # module for exporing Guassian kernel function and derivatives
export R, gradR, g1g1R, g1g2R

#---------------------------------------
# kernel evals and derivatives 
#------------------------------------
r(d::Real,sigma) = exp(-0.5 * d * d / (sigma * sigma))

function rp_div_d(d::Real,sigma) 
    s2 = sigma * sigma
    -exp(-0.5 * d * d / s2) / s2
end

function rpp(d::Real,sigma)
    rd = r(d,sigma)
    s2 = sigma * sigma
    d * d * rd / (s2 * s2) - rd / s2
end

function R{T<:Real}(x::Array{T,1},y::Array{T,1},sigma)
    r(norm(x-y),sigma)
end

function gradR{T<:Real}(x::Array{T,1},y::Array{T,1},sigma)
    v=x-y
    n=norm(v)
    v * rp_div_d(n,sigma)
end

outer(u,v) = u*transpose(v)

function g1g2R(x, y, sigma)
    v = x-y
    n = norm(v)
    u = v/n
    eey = eye(length(x))
    rpd = rp_div_d(n,sigma)
    if n == 0
        G = -rpp(n,sigma) * eey 
        return G
    else
        G = -rpd * eey
        G += outer(u,-u) * (rpp(n,sigma) - rpd) 
        return G
    end
end

g1g1R(x, y,sigma) = -g1g2R(x, y,sigma)

end
