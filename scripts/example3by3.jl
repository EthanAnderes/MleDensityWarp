#=
include("scripts/example3by3.jl")
=#
addprocs(4)
using WarpFlow
using PyPlot

#---- set the data and the target
dim = 2
tmpx = shuffle([rand(50); randn(80)/10 .+ .8])
tmpy = tmpx + rand(size(tmpx)) .* 1.1
nPhi = length(tmpx)
X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx), (tmpy[i]-minimum(tmpy))/maximum(tmpy)] for i=1:nPhi]
# target(x) = targetUnif2d(x; targSig = 0.1, limit = 0.4, center = 0.5)
@everywhere target(x) = WarpFlow.targetNormal2d(x; targSig = 0.5, center = [0.5,0.5])


# set an array of lambda, sigmas
matSigmaLambda = [(s,l) for s in (0.2, 0.1), l in (5.0, 0.5)]

#  gradient ascent on kappa and eta_coeff
est_den_container = @parallel (hcat) for (s,l) in matSigmaLambda
	kappa = Array{Float64,1}[X[Int(i)]  for i in 1:round(nPhi/2)]
	y0    = Flow(kappa, WarpFlow.array1(dim, length(kappa)), X, WarpFlow.array2eye(dim, nPhi), dim)
	for counter = 1:300
		z0 = WarpFlow.get_grad(y0, target, l, s)
		y0 = y0 + 0.005 * z0
	end
	#-- now we save the estimated density
	x_grd, y_grd =  WarpFlow.meshgrid(linspace(-0.1, 1.1, 175), linspace(-0.1, 1.1, 175))
	phix_grd_0   = Array{Float64,1}[[x_grd[i]; y_grd[i]] for i=1:length(x_grd)]
	Dphix_grd_0  = Array{Float64,2}[eye(2)               for i=1:length(x_grd)]
	yplt0        = Flow(y0.kappa, y0.eta_coeff, phix_grd_0, Dphix_grd_0, dim)
	dydt(t,y)    = WarpFlow.d_forward_dt(y, s)
 	(t1, yplt1)  = WarpFlow.ode23_abstract_end(dydt, [0,1], yplt0) # Flow y0 forward to time 1
	det_grd = Float64[abs(det(yplt1.Dphix[i])) for i=1:length(x_grd)]
	den,    = target(yplt1.phix)
	det_grd .* den
end

rmprocs(workers())

# writecsv("simulations/example3by3den.csv", est_den_container)
# writecsv("simulations/example3by3X.csv", hcat(X...))

#dim = 2
#matSigmaLambda = [(s,l) for s in (0.4, 0.2, 0.05), l in (10.0, 5.0, 1.0)]
#est_den_container = readcsv("simulations/example3by3den.csv")
#Xcat = readcsv("simulations/example3by3X.csv")
#X =  Array{Float64,1}[Xcat[:,k] for k in 1:size(Xcat,2) ]
x_grd, y_grd =  WarpFlow.meshgrid(linspace(-0.1, 1.1, 175),linspace(-0.1, 1.1, 175))
fig = figure(figsize=(14,14))
for k = 1:4
	subplot(2,2,k)
	title("(sigma, lambda) = $(matSigmaLambda[k])")
	scatter(Float64[point[1] for point in X], Float64[point[2] for point in X], c="b")
	contour(x_grd, y_grd, reshape(est_den_container[:,k], size(x_grd)), 30)
end
# plt[:savefig]("simulations/example3by3.pdf",dpi=180)
# plt[:close](fig)
