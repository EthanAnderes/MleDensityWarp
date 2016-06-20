#=
push!(LOAD_PATH, "<path to directory WarpFlow>") # <--- use this if Julia can't find WarpFlow
include("scripts/example1.jl")
=#
using WarpFlow
using PyPlot

# set the data and the target
dim = 1
tmpx = shuffle([rand(45); randn(175)/10 .+ .8])
nPhi = length(tmpx)
X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx)] for i=1:nPhi]

# set the target function by overwriting the default...for v0.5 just pass a function.
target(x) = WarpFlow.targetUnif1d(x; targSig = 0.1, limit = 0.35, center = 0.5) # this sets an alias to a function giving in the file targets.jl


# generate Flow object
kappa  = Array{Float64,1}[X[Int(i)]  for i in 1:round(nPhi/4)]
nKap   = length(kappa)
y0     = Flow(kappa, WarpFlow.array1(dim, nKap), X, WarpFlow.array2eye(dim, nPhi), dim)

# function to save images
function plotrun(;fignum = 1, save = false)
	x_grd = linspace(-0.2, 1.2, 300)
	phix_grd_0  = Array{Float64,1}[[x_grd[i]] for i=1:length(x_grd)]
	Dphix_grd_0 = Array{Float64,2}[ones(1,1)  for i=1:length(x_grd)]
	yplt0 = Flow(y0.kappa, y0.eta_coeff, phix_grd_0, Dphix_grd_0, dim)
	dydt(t,y)= WarpFlow.d_forward_dt(y, sigma)
	(t1,yplt1)=WarpFlow.ode23_abstract_end(dydt,[0,1], yplt0) # Flow y0 forward to time 1
	det_grd = Float64[abs(yplt1.Dphix[i][1]) for i=1:length(x_grd)]
	den,    = target(yplt1.phix)
	est_den = det_grd.*den

	fig = figure()
	plt[:hist]([pnt[1] for pnt in X], 50, normed=1, histtype="stepfilled")
	plot(x_grd, est_den)
	if save
		plt[:savefig]("simulations/example1_v$fignum.pdf",dpi=180)
		plt[:close](fig)
	end
	return nothing
end


#  gradient ascent on eta_coeff
lambda, sigma = 1.0, 0.1
for counter = 1:35
	@time z0 = WarpFlow.get_grad(y0, target, lambda, sigma)
	y0 = y0 + 0.002 * z0
end

plotrun()

#
# #  gradient ascent on eta_coeff
# lambda, sigma = 0.1, 0.05
# for counter = 1:35
# 	@time z0 = get_grad(y0, lambda, sigma)
# 	y0 = y0 + 0.002 * z0
# end
#
# plotrun()
