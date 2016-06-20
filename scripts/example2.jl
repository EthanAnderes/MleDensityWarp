#=
push!(LOAD_PATH, "<path to directory WarpFlow>") # <--- use this if Julia can't find WarpFlow
include("scripts/example2.jl")
=#
using WarpFlow
using PyPlot

# set the data and the target
dim = 2
tmpx = shuffle([rand(50); randn(80)/10 .+ .8])
tmpy = tmpx + rand(size(tmpx)) .* 1.1
nPhi = length(tmpx)
X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx), (tmpy[i]-minimum(tmpy))/maximum(tmpy)] for i=1:nPhi]
# target(x) = targetUnif2d(x; targSig = 0.1, limit = 0.4, center = 0.5)
target(x) = WarpFlow.targetNormal2d(x; targSig = 0.5, center = [0.5,0.5])

# generate Flow object
kappa     = Array{Float64,1}[X[Int(i)]  for i in 1:round(nPhi/2)]
# x_grd_kns, y_grd_kns =  meshgrid(linspace(-1.0,1.0, 10),linspace(-1.0,1.0, 10))
# append!(kappa, Array{Float64,1}[ [x_grd_kns[i], y_grd_kns[i]] for i in 1:length(x_grd_kns)])
nKap   = length(kappa)

y0 = Flow(kappa, WarpFlow.array1(dim, nKap), X, WarpFlow.array2eye(dim, nPhi), dim)


# function to save images
function plotrun(;fignum = 1, save = false)
	x_grd, y_grd =  meshgrid(linspace(-0.1, 1.1, 175),linspace(-0.1, 1.1, 175))
	phix_grd_0   = Array{Float64,1}[[x_grd[i], y_grd[i]] for i=1:length(x_grd)]
	Dphix_grd_0  = Array{Float64,2}[eye(2)               for i=1:length(x_grd)]
	yplt0        = WarpFlow.Flow(y0.kappa, y0.eta_coeff, phix_grd_0, Dphix_grd_0, dim)
	dydt(t,y)    = WarpFlow.d_forward_dt(y, sigma)
	(t1,yplt1)   = WarpFlow.ode23_abstract_end(dydt,[0,1], yplt0) # Flow y0 forward to time 1
	det_grd  = Float64[abs(det(yplt1.Dphix[i])) for i=1:length(x_grd)]
	den,     = target(yplt1.phix)
	est_den  = det_grd.*den

	fig = figure()
	scatter(Float64[point[1] for point in X], Float64[point[2] for point in X], c="b")
	contour(x_grd, y_grd, reshape(est_den,size(x_grd)), 30 )
	if save
		plt[:savefig]("scripts/example2_v$fignum.pdf",dpi=180)
		plt[:close](fig)
	end
end


#  gradient ascent on kappa and eta_coeff
lambda, sigma = 1.0, 0.1
for counter = 1:50
	@time z0 = WarpFlow.get_grad(y0, target, lambda, sigma)
	y0 = y0 + 0.005 * z0
end

plotrun()
