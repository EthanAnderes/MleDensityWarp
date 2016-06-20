module WarpFlow # module for exporing Guassian kernel function and derivatives

export Flow, TrFlow



#############################################

# Custom Types

################################################

typealias Array1 Array{Array{Float64,1},1} # for lists of vectors

typealias Array2 Array{Array{Float64,2},1} # for lists of jacobians

abstract AbstractFlow

immutable Flow <: AbstractFlow
	kappa::Array1
	eta_coeff::Array1
	phix::Array1
	Dphix::Array2
	dim::Int64
end

immutable TrFlow <: AbstractFlow
	kappa::Array1
	eta_coeff::Array1
	phix::Array1
	Dphix::Array2
	dlkappa::Array1
	dleta_coeff::Array1
	dlphix::Array1
	dlDphix::Array2
	dim::Int64
end


########################################

# type constructors

##################################


# zero Flow constructor
Flow(dim, nKap, nPhi) = Flow(array1(dim, nKap), array1(dim, nKap), array1(dim, nPhi), array2(dim, nPhi), dim)

# zero TrFlow constructor
TrFlow(dim, nKap, nPhi) = TrFlow(array1(dim, nKap), array1(dim, nKap), array1(dim, nPhi), array2(dim, nPhi), array1(dim, nKap), array1(dim, nKap), array1(dim, nPhi), array2(dim, nPhi), dim) # zero Flow constructor

# this is used to initialize the transpose flow at time 1 in get_grad
function TrFlow(yin::Flow, target::Function)
	nKap = length(yin.kappa)
	nPhi = length(yin.phix)
	dleta_coeff = array1(yin.dim, nKap)
	dlkappa     = array1(yin.dim, nKap)
	dlphix      = Array{Float64,1}[target(yin.phix)[2][i]/nPhi for i = 1:nPhi]
	dlDphix     = Array{Float64,2}[inv(yin.Dphix[i])/nPhi for i = 1:nPhi]
	TrFlow(yin.kappa, yin.eta_coeff, yin.phix, yin.Dphix, dlkappa, dleta_coeff, dlphix, dlDphix, yin.dim)
end


array1(dim, k) = Array{Float64,1}[zeros(dim) for i=1:k] # constructor for Array1

array2(dim, k) = Array{Float64,2}[zeros(dim, dim) for i=1:k] # constructor for Array2

array2eye(dim, k) = Array{Float64,2}[eye(dim) for i=1:k] # constructor for Array2


######################################################

#  extend basic functions to Flow types

######################################################

import Base: +, -, *, /

# defines + between Flow and TrFlow... used to update with  the return value of get_grad
function +(yin::Flow, zin::TrFlow)
	nKap = length(yin.kappa) # TODO: thro an error if dims don't match
	nPhi = length(yin.phix)
	yout = Flow(yin.dim, 0, 0) # initialize an empty Flow
	for i = 1:nKap
		push!(yout.kappa,     yin.kappa[i]     + zin.dlkappa[i])
		push!(yout.eta_coeff, yin.eta_coeff[i] + zin.dleta_coeff[i])
	end
	# phix and Dphix don't get updated from TrFlow
	for i = 1:nPhi
		push!(yout.phix,   yin.phix[i])
		push!(yout.Dphix,  yin.Dphix[i])
	end
	yout
end

# defines +, -, *, / between Flow and Flow, used in ode
for opt = (:+, :-, :*, :/)
	@eval begin
		function $opt(yin::Flow, zin::Flow)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = Flow(yin.dim, 0, 0)
			for i = 1:nKap
				push!(yout.kappa,     $opt(yin.kappa[i], zin.kappa[i]))
				push!(yout.eta_coeff, $opt(yin.eta_coeff[i], zin.eta_coeff[i]))
			end
			for i = 1:nPhi
				push!(yout.phix,  $opt(yin.phix[i],  zin.phix[i]))
				push!(yout.Dphix, $opt(yin.Dphix[i], zin.Dphix[i]))
			end
			yout
		end # end function definition
	end
end

# defines +, -, *, / between TrFlow and TrFlow, used in ode
for opt = (:+, :-, :*, :/)
	@eval begin
		function $opt(yin::TrFlow, zin::TrFlow)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = TrFlow(yin.dim, 0, 0)
			for i = 1:nKap
				push!(yout.kappa,       $opt(yin.kappa[i], zin.kappa[i]))
				push!(yout.eta_coeff,   $opt(yin.eta_coeff[i], zin.eta_coeff[i]))
				push!(yout.dlkappa,     $opt(yin.dlkappa[i], zin.dlkappa[i]))
				push!(yout.dleta_coeff, $opt(yin.dleta_coeff[i], zin.dleta_coeff[i]))
			end
			for i = 1:nPhi
				push!(yout.phix,   $opt(yin.phix[i], zin.phix[i]))
				push!(yout.Dphix,  $opt(yin.Dphix[i], zin.Dphix[i]))
				push!(yout.dlphix, $opt(yin.dlphix[i], zin.dlphix[i]))
				push!(yout.dlDphix,$opt(yin.dlDphix[i], zin.dlDphix[i]))
			end
			yout
		end # end function definition
	end
end

# defines +, -, *, / between Flow and Float64
for opt = (:+, :-, :*, :/)
	@eval begin
		function $opt(yin::Flow, a::Real)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = Flow(yin.dim, 0, 0)
			for i = 1:nKap
				push!(yout.kappa, $opt(yin.kappa[i],a))
				push!(yout.eta_coeff, $opt(yin.eta_coeff[i],a))
			end
			for i = 1:nPhi
				push!(yout.phix, $opt(yin.phix[i],a))
				push!(yout.Dphix, $opt(yin.Dphix[i],a))
			end
			yout
		end # end function definition
	end
end

# defines +, -, *, / between TrFlow and Float64
for opt = (:+, :-, :*, :/)
	@eval begin
		function $opt(yin::TrFlow, a::Real)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = TrFlow(yin.dim, 0, 0)
			for i = 1:nKap
				push!(yout.kappa, $opt(yin.kappa[i],a))
				push!(yout.eta_coeff, $opt(yin.eta_coeff[i],a))
				push!(yout.dlkappa, $opt(yin.dlkappa[i],a))
				push!(yout.dleta_coeff, $opt(yin.dleta_coeff[i],a))
			end
			for i = 1:nPhi
				push!(yout.phix, $opt(yin.phix[i],a))
				push!(yout.Dphix, $opt(yin.Dphix[i],a))
				push!(yout.dlphix, $opt(yin.dlphix[i],a))
				push!(yout.dlDphix, $opt(yin.dlDphix[i],a))
			end
			yout
		end # end function definition
	end
end

# +, * are associative for both Flow and TrFlow
for opt = (:+, :*)
	@eval $opt{T<:AbstractFlow}(a::Real, yin::T) = $opt(yin, a)
end

-{T<:AbstractFlow}(a::Real, yin::T) = -1.0 * -(yin, a)



####################################################

# code for extending ode algorithms to Flows and TrFlows

#####################################################


function get_grad(y0::Flow, target::Function, lambda, sigma)
    dydt(t,y) = d_forward_dt(y, sigma)
    (t1,y1) = ode23_abstract_end(dydt, [0,1], y0) # Flow y0 forward to time 1

    z1 = TrFlow(y1, target) # Initalize the transpose flow at time 1

    dzdt(t,z) = d_backward_dt(z, sigma)
    (t0,z0) = ode23_abstract_end(dzdt, [0,1], z1) # Flow z1 backward from time 1 to 0

    # add the regularization
    nKap = length(y0.kappa)
    for i = 1:nKap, j = 1:nKap
        z0.dleta_coeff[i] -= lambda * y0.eta_coeff[j] * R(y0.kappa[i], y0.kappa[j], sigma)
        z0.dlkappa[i]     -= lambda * dot(y0.eta_coeff[i], y0.eta_coeff[j]) * gradR(y0.kappa[i], y0.kappa[j], sigma)
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
    gradRkk = Array{Float64,1}[WarpFlow.gradR(yin.kappa[i], yin.kappa[j],sigma) for i = 1:nKap, j = 1:nKap]
    gradRpk = Array{Float64,1}[WarpFlow.gradR(yin.phix[i], yin.kappa[j],sigma) for i = 1:nPhi, j = 1:nKap]
    Rkk = Float64[WarpFlow.R(yin.kappa[i], yin.kappa[j],sigma) for i = 1:nKap, j = 1:nKap]
    Rpk = Float64[WarpFlow.R(yin.phix[i], yin.kappa[j],sigma) for i = 1:nPhi, j = 1:nKap]
    g1g2Rkkdl = Array{Float64,1}[WarpFlow.g1g2R(yin.kappa[i],yin.kappa[j], sigma)*yin.dleta_coeff[i] for i = 1:nKap, j = 1:nKap]# note g1g1Rkk = - g1g2Rkk
    g1g1RpkD  = Array{Float64,1}[WarpFlow.g1g1R(yin.phix[i],yin.kappa[j],sigma)*yin.Dphix[i][:,col]*transpose(yin.eta_coeff[j])*yin.dlDphix[i][:,col]  for i = 1:nPhi, j = 1:nKap, col = 1:yin.dim]

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
            yout.dlphix[i] +=  g1g1RpkD[i,j,col] #g1g1R(yin.phix[i], yin.kappa[j], sigma) * yin.Dphix[i][:,col] * transpose(yin.eta_coeff[j]) * yin.dlDphix[i][:,col]
            yout.dlDphix[i][:,col] +=  gradRpk[i,j] * transpose(yin.eta_coeff[j]) * yin.dlDphix[i][:,col]
        end
    end
    # dlkappa
    for i = 1:nKap, j = 1:nKap
        yout.dlkappa[i] += dot(yin.eta_coeff[i], yin.eta_coeff[j]) .* g1g2Rkkdl[i,j] #(g1g1R(yin.kappa[i], yin.kappa[j], sigma) * yin.dleta_coeff[i])
        yout.dlkappa[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) .* g1g2Rkkdl[j,i] #(g1g2R(yin.kappa[j], yin.kappa[i], sigma) * yin.dleta_coeff[j])
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
            yout.dlkappa[i] -= g1g1RpkD[j,i,col] # -g1g2R(yin.phix[j], yin.kappa[i],sigma) * yin.Dphix[j][:,col] * transpose(yin.eta_coeff[i]) * yin.dlDphix[j][:,col]
            yout.dleta_coeff[i] += dot(yin.Dphix[j][:,col], gradRpk[j,i]) * yin.dlDphix[j][:,col]
        end
    end
    yout
end



function ode23_abstract_end(F::Function, tspan::AbstractVector, y0)
    rtol = 1.e-4
    atol = 1.e-6
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
        s2 = F(t+h/2.0, y+(h/2.0)*s1)
        s3 = F(t+3.0*h/4.0, y + (3.0*h/4.0)*s2)
        tnew = t + h
        ynew = y + (h*2.0/9.0)*s1 + (h*3.0/9.0)*s2 + (h*4.0/9.0)*s3
        s4 = F(tnew, ynew)

        # Estimate the error.
        e = (-h*5.0/72.0)*s1 + (h*6.0/72.0)*s2 + (h*8.0/72.0)*s3 + (-h*9.0/72.0)*s4
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





# used in ode
function norm_inf_any{D}(y::Array{Float64,D})
    norm(y,Inf)
end

# used in ode
function norm_inf_any(y::Flow)
    nKap = length(y.kappa)
    nPhi = length(y.phix)
    m=-Inf
    for i=1:nKap
        m = max(norm_inf_any(y.kappa[i]), m)
        m = max(norm_inf_any(y.eta_coeff[i]), m)
    end
    for i=1:nPhi
        m = max(norm_inf_any(y.phix[i]), m)
        m = max(norm_inf_any(y.Dphix[i]), m)
    end
    m
end

# used in ode
function norm_inf_any(y::TrFlow)
    nKap = length(y.kappa)
    nPhi = length(y.phix)
    m=-Inf
    for i=1:nKap
        m = max(norm_inf_any(y.kappa[i]), m)
        m = max(norm_inf_any(y.eta_coeff[i]), m)
        m = max(norm_inf_any(y.dlkappa[i]), m)
        m = max(norm_inf_any(y.dleta_coeff[i]), m)
    end
    for i=1:nPhi
        m = max(norm_inf_any(y.phix[i]), m)
        m = max(norm_inf_any(y.Dphix[i]), m)
        m = max(norm_inf_any(y.dlphix[i]), m)
        m = max(norm_inf_any(y.dlDphix[i]), m)
    end
    m
end


#############################################

# Kernels for the spline flows

#################################################



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





#################################################

# Target densities

###################################################

#------------------------------------
#  Uniform targets
#--------------------------------------------
function targetUnif1d(kappa::Array{Array{Float64,1},1}; targSig = 0.1, limit = 0.4, center = 0.0)
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k =1:length(kappa)
		kap = kappa[k][1]
		if kap <= center-limit
			dentmp = exp(-((kap  - (center - limit)).^2.0)/(2.0 * targSig * targSig))
			gradHtmp = -(kap - (center - limit)) / (targSig.^2.0)
		elseif center-limit < kap < center+limit
			dentmp = 1.0
			gradHtmp = 0.0
		else
			dentmp = exp(-((kap - (center + limit)).^2.0)/(2.0*targSig.^2.0))
			gradHtmp = -(kap - (center + limit)) / (targSig.^2.0)
		end
		push!(density, dentmp / (limit + limit + targSig*sqrt(2.0*pi))  )
		push!(gradH,[gradHtmp])
	end
	density, gradH
end
function targetUnif2d(kappa::Array{Array{Float64,1},1}; targSig = 0.1, limit = 0.4, center = 0.0)
	N = length(kappa)
	densitx, gradHx = targetUnif1d(Array{Float64,1}[[kappa[i][1]] for i=1:N]; targSig = targSig, limit = limit, center = center)
	density, gradHy = targetUnif1d(Array{Float64,1}[[kappa[i][2]] for i=1:N]; targSig = targSig, limit = limit, center = center)
	densitx .* density, Array{Float64,1}[[gradHx[i][1], gradHy[i][1]] for i=1:N]
end
function targetUnif3d(kappa::Array{Array{Float64,1},1}; targSig = 0.1, limit = 0.4, center = 0.0)
	N = length(kappa)
	densitx, gradHx = targetUnif1d(Array{Float64,1}[[kappa[i][1]] for i=1:N]; targSig = targSig, limit = limit, center = center)
	density, gradHy = targetUnif1d(Array{Float64,1}[[kappa[i][2]] for i=1:N]; targSig = targSig, limit = limit, center = center)
	densitz, gradHz = targetUnif1d(Array{Float64,1}[[kappa[i][2]] for i=1:N]; targSig = targSig, limit = limit, center = center)
	densitx .* density .* densitz, Array{Float64,1}[[gradHx[i][1], gradHy[i][1], gradHz[i][1]] for i=1:N]
end


#------------------------------------
#  Normal Targets
#--------------------------------------------
function targetNormal1d(kappa::Array{Array{Float64,1},1}; targSig = 0.1, center = 0.0)
	density = Float64[]
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k = 1:length(kappa)
		push!(density, exp(- (kappa[k][1] - center).^2 / (2.0 *targSig^2) ) / (sqrt(2.0*pi) * targSig) )
		push!(gradH,[-(kappa[k][1]-center)/(targSig^2)])
	end
	density, gradH
end
function targetNormal2d(kappa::Array{Array{Float64,1},1}; targSig = 0.1, center = [0.0,0.0])
	N = length(kappa)
	densitx, gradHx = targetNormal1d(Array{Float64,1}[[kappa[i][1]] for i=1:N], targSig = targSig, center = center[1])
	density, gradHy = targetNormal1d(Array{Float64,1}[[kappa[i][2]] for i=1:N], targSig = targSig, center = center[2])
	densitx .* density, Array{Float64,1}[[gradHx[i][1], gradHy[i][1]] for i=1:N]
end
function targetNormal3d(kappa::Array{Array{Float64,1},1}; targSig = 0.1, center = [0.0,0.0,0.0])
	N = length(kappa)
	densitx, gradHx = targetNormal1d(Array{Float64,1}[[kappa[i][1]] for i=1:N], targSig = targSig, center = center[1])
	density, gradHy = targetNormal1d(Array{Float64,1}[[kappa[i][2]] for i=1:N], targSig = targSig, center = center[2])
	densitz, gradHz = targetNormal1d(Array{Float64,1}[[kappa[i][2]] for i=1:N], targSig = targSig, center = center[3])
	densitx .* density .* densitz, Array{Float64,1}[[gradHx[i][1], gradHy[i][1], gradHz[i][1]] for i=1:N]
end


#------------------------------------
# The following is old code and needs to be ported to Julia
#--------------------------------------------
function targetGaussianMixture2d(x, p,mu1,mu2,sigma1,sigma2)
	den=  p*normpdf(x,mu1,sigma1) + (1-p)*normpdf(x,mu2,sigma2)
	dden1=(  p)*normpdf(x,mu1,sigma1).*(-(x-mu1)/sigma1^2)
	dden2=(1-p)*normpdf(x,mu2,sigma2).*(-(x-mu2)/sigma2^2)
	# dden=den.*(-(kap-mu)/TargSig^2)
	gradH=(dden1+dden2)./den
	den, gradH
end

function hfun12d(DefXnew,DefYnew)
	n=2
	sigma=.5
	gradx=(-n/2)*(1/sigma).*(2*DefXnew/sigma)./(1+(DefXnew/sigma).^2)
	grady=(-n/2)*(1/sigma).*(2*DefYnew/sigma)./(1+(DefYnew/sigma).^2)

	expH=(1/sigma^2)*(gamma(n/2)/(sqrt(pi)*gamma((n-1)/2)))^2 * (1+(DefXnew/sigma).^2).^(-n/2).*(1+(DefYnew/sigma).^2).^(-n/2)
	expH, gradx, grady
end
function hfun22d(DefXnew,DefYnew)
	n=2
	sigma=3
	sigma2=1
	gradx=(-n/2)*(1/sigma).*(2*DefXnew/sigma)./(1+(DefXnew/sigma).^2)
	grady=(-1/sigma2^2).*(DefYnew)

	expH=(1/sigma)*(gamma(n/2)/(sqrt(pi)*gamma((n-1)/2))) * (1+(DefXnew/sigma).^2).^(-n/2).*(1/(sqrt(2*pi)*sigma2)).*(exp(-DefYnew.^2/(2*sigma^2)))
	expH, gradx, grady
end





#######################################################

#  misc functions

#######################################################

function meshgrid(side_x,side_y)
    x = repmat(reshape([side_x],(1,length(side_x))) ,length(side_y),1)
    y = repmat(reshape([side_y],(length(side_y),1)) ,1,length(side_x))
    x,y
end


end # module
