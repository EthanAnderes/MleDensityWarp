include("../src/flow_ode.jl")
include("../src/rfuncs.jl")
include("../src/targets.jl")
using  PyCall
@pyimport matplotlib.pyplot as plt 

# generate the data:  X and set the target measure
tmpx = [rand(45), randn(75)/10 + .8]
N = length(tmpx)
X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx)] for i=1:N]
target(x) = targetUnif1d(x; targSig = 0.1, limit = 0.35, center = 0.5) # this sets an alias to a function giving in the file targets.jl
kappa     = Array{Float64,1}[X[i]  for i in 1:round(N)]
n_knots   = length(kappa)
eta_coeff = Array{Float64,1}[zero(kappa[i]) for i in 1:n_knots]


function saveim(fignum)
	x_grd = linspace(-0.2,1.2, 350)  
	phix_grd_0  = Array{Float64,1}[[x_grd[i]] for i=1:length(x_grd)]
	
	phix_grd_1, Dphix_grd_1 = forward_flow(lambda_sigma[2], kappa, eta_coeff, phix_grd_0, Array{Float64,1}[[1.0] for i in 1:length(x_grd)])
	det_grd = Float64[abs(Dphix_grd_1[i][1]) for i=1:length(x_grd)]
	den, placeholder = target(phix_grd_0)
	est_den = det_grd.*den
	
	fig = plt.figure()
	plt.hist([pnt[1] for pnt in X], 50, normed=1, histtype="stepfilled")
	plt.plot(x_grd, est_den)
	plt.savefig("out/example1_v$fignum.pdf",dpi=180)
	plt.close(fig)
end


#  gradient ascent on eta_coeff
lambda_sigma = [5.0, 0.1] 
for counter = 1:30
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,1}[[1.0] for i in 1:N])
	eta_coeff += prodc(0.002, dleta_coeff)
	toc()
end
saveim(1)



#  gradient ascent on eta_coeff
lambda_sigma = [1.0, 0.1] 
for counter = 1:30
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,1}[[1.0] for i in 1:N])
	eta_coeff += prodc(0.002, dleta_coeff)
	toc()
end
saveim(2)




#  gradient ascent on eta_coeff
lambda_sigma = [0.1, 0.1] 
for counter = 1:30
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,1}[[1.0] for i in 1:N])
	eta_coeff += prodc(0.002, dleta_coeff)
	toc()
end
saveim(3)

