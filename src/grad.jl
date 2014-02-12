
function get_grad(y0::Flow, lambda, sigma)
    dydt(t,y) = d_forward_dt(y, sigma)
    (t1,y1) = ode23_abstract_end(dydt, [0,1], y0) # Flow y0 forward to time 1
    
    z1 = TrFlow(y1, target) # Initalize the transpose flow at time 1
   
    dzdt(t,z) = d_backward_dt(z, sigma)
    (t0,z0) = ode23_abstract_end(dzdt, [0,1], z1) # Flow z1 backward from time 1 to 0 

    # add the regularization
    nKap = length(y0.kappa)
    for i = 1:nKap, j = 1:nKap 
        z0.dleta_coeff[i] -= lambda * y0.eta_coeff[j] * R(y0.kappa[i], y0.kappa[j], sigma) 
    end
    # The next for loop is where we should add the kappa regularization.
    # Since we haven't programed it, I'm setting the kappa updates to zero.
    for i = 1:nKap 
        z0.dlkappa[i] = 0.0 * z0.dlkappa[i]
    end
    z0 # return the TrFlow
end



function d_forward_dt(yin::Flow, sigma)
    nKap = length(yin.kappa)
    nPhi = length(yin.phix)
    yout = Flow(yin.dim, nKap, nPhi) # intialize the return Flow
    # eta and kappa
    for i = 1:nKap, j = 1:nKap 
        yout.eta_coeff[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) * gradR(yin.kappa[i], yin.kappa[j], sigma) 
        yout.kappa[i] += yin.eta_coeff[j] * R(yin.kappa[i], yin.kappa[j], sigma) 
    end
    # phi
    for i = 1:nPhi, j = 1:nKap 
        yout.phix[i] += yin.eta_coeff[j] * R(yin.phix[i], yin.kappa[j], sigma) 
        yout.Dphix[i] += yin.eta_coeff[j] * transpose(gradR(yin.phix[i], yin.kappa[j], sigma)) * yin.Dphix[i]
    end
    yout
end



function d_backward_dt(yin::TrFlow, sigma)
    nKap = length(yin.kappa)
    nPhi = length(yin.phix)
    yout = TrFlow(yin.dim, nKap, nPhi)
    
    # the following are re-used 
    gradRkk = Array{Float64,1}[gradR(yin.kappa[i], yin.kappa[j],sigma) for i = 1:nKap, j = 1:nKap]
    gradRpk = Array{Float64,1}[gradR(yin.phix[i], yin.kappa[j],sigma) for i = 1:nPhi, j = 1:nKap]
    Rkk = Float64[R(yin.kappa[i], yin.kappa[j],sigma) for i = 1:nKap, j = 1:nKap]
    Rpk = Float64[R(yin.phix[i], yin.kappa[j],sigma) for i = 1:nPhi, j = 1:nKap]

    # eta and kappa
    for i = 1:nKap, j = 1:nKap 
        yout.eta_coeff[i] += dot(yin.eta_coeff[i], yin.eta_coeff[j]) * gradRkk[i,j] 
        yout.kappa[i] -= yin.eta_coeff[j] * Rkk[i,j]
    end
    # phi and Dphi
    for i = 1:nPhi, j = 1:nKap 
        yout.phix[i] -= yin.eta_coeff[j] * Rpk[i,j]
        yout.Dphix[i] -= yin.eta_coeff[j] * transpose(gradRpk[i,j]) * yin.Dphix[i]
    end
    # dlphix and dlDphix
    for i = 1:nPhi, j = 1:nKap
        yout.dlphix[i] += gradRpk[i,j] * transpose(yin.eta_coeff[j]) * yin.dlphix[i]
        for col = 1:yin.dim
            yout.dlphix[i] +=  g1g1R(yin.phix[i], yin.kappa[j], sigma) * yin.Dphix[i][:,col] * transpose(yin.eta_coeff[j]) * yin.dlDphix[i][:,col]
            yout.dlDphix[i][:,col] +=  gradRpk[i,j] * transpose(yin.eta_coeff[j]) * yin.dlDphix[i][:,col]
        end
    end
    # dlkappa
    for i = 1:nKap, j = 1:nKap
        yout.dlkappa[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) .* (g1g1R(yin.kappa[i], yin.kappa[j], sigma) * yin.dleta_coeff[i])
        yout.dlkappa[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) .* (g1g2R(yin.kappa[j], yin.kappa[i], sigma) * yin.dleta_coeff[j])
        yout.dlkappa[i] += gradRkk[i,j] * transpose(yin.eta_coeff[j]) * yin.dlkappa[i]
        yout.dlkappa[i] -= gradRkk[j,i] * transpose(yin.eta_coeff[i]) * yin.dlkappa[j]
        yout.dleta_coeff[i] -= yin.eta_coeff[j] * transpose(gradRkk[i,j]) * yin.dleta_coeff[i]
        yout.dleta_coeff[i] -= yin.eta_coeff[j] * transpose(gradRkk[j,i]) * yin.dleta_coeff[j]
        yout.dleta_coeff[i] += Rkk[j,i] * yin.dlkappa[j]
    end
    for i = 1:nKap, j = 1:nPhi
        yout.dlkappa[i] -= gradRpk[j,i] * transpose(yin.eta_coeff[i]) * yin.dlphix[j]
        yout.dleta_coeff[i] += Rpk[j,i] * yin.dlphix[j]
        for col = 1:yin.dim
            yout.dlkappa[i] += g1g2R(yin.phix[j], yin.kappa[i],sigma) * yin.Dphix[j][:,col] * transpose(yin.eta_coeff[i]) * yin.dlDphix[j][:,col]
            yout.dleta_coeff[i] += dot(yin.Dphix[j][:,col], gradRpk[j,i]) * yin.dlDphix[j][:,col]
        end
    end 
    yout
end



function ode23_abstract_end(F::Function, tspan::AbstractVector, y0)
    rtol = 1.e-5
    atol = 1.e-8
    threshold = atol / rtol

    t = tspan[1]
    tfinal = tspan[end]
    tdir = sign(tfinal - t)
    hmax = abs(0.1*(tfinal-t))
    y = y0
    tlen = length(t)

    # Compute initial step size.
    s1 = F(t, y)
    r = norm_inf_any(s1/max(norm_inf_any(y), threshold)) + realmin() 
    h = tdir*0.8*rtol^(1/3)/r
    
    while t != tfinal  # The main loop.
        hmin = 16*eps()*abs(t)
        if abs(h) > hmax; h = tdir*hmax; end
        if abs(h) < hmin; h = tdir*hmin; end

        # Stretch the step if t is close to tfinal.
        if 1.1*abs(h) >= abs(tfinal - t)
            h = tfinal - t
        end

        # Attempt a step.
        s2 = F(t+h/2.0, y+h/2.0*s1)
        s3 = F(t+3.0*h/4.0, y+3.0*h/4.0*s2)
        tnew = t + h
        ynew = y + h*(2.0*s1 + 3.0*s2 + 4.0*s3)/9.0
        s4 = F(tnew, ynew)

        # Estimate the error.
        e = h*(-5.0*s1 + 6.0*s2 + 8.0*s3 - 9.0*s4)/72.0
        ##err = norm(e./max(max(abs(y), abs(ynew)), threshold), Inf) + realmin()
        err = norm_inf_any(e)/max(max(norm_inf_any(y), norm_inf_any(ynew)), threshold) + realmin()
        # Accept the solution if the estimated error is less than the tolerance.
        if err <= rtol
            t = tnew
            y = ynew
            s1 = s4   # Reuse final function value to start new step
        end

        # Compute a new step size.
        h = h*min(5.0, 0.8*(rtol/err)^(1/3))

        # Exit early if step size is too small.
        if abs(h) <= hmin
            println("Step size ", h, " too small at t = ", t)
            t = tfinal
        end
    end # while (t != tfinal)
    return (t, y)
end # ode23_abstract_end




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

g1g1R(x, y,sigma) = -g1g2R(x,y,sigma)


#------------------------------------
# misc functions
#--------------------------------------------
function meshgrid(side_x,side_y)
    x = repmat(reshape([side_x],(1,length(side_x))) ,length(side_y),1)
    y = repmat(reshape([side_y],(length(side_y),1)) ,1,length(side_x))
    x,y
end


