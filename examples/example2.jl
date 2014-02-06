include("../src/flow_ode.jl")
include("../src/rfuncs.jl")
include("../src/targets.jl")
using  PyCall
@pyimport matplotlib.pyplot as plt 

tmpx = [rand(50), randn(70)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)

X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx), (tmpy[i]-minimum(tmpy))/maximum(tmpy)] for i=1:N]
target(x) = targetUnif2d(x; targSig = 0.1, limit = 0.25, center = 0.5) # this sets an alias to a function giving in the file targets.jl
kappa     = Array{Float64,1}[X[i]  for i in 1:round(N/2)]
n_knots   = length(kappa)
eta_coeff = Array{Float64,1}[zero(kappa[i]) for i in 1:n_knots]

function saveim(fignum)
	x_grd, y_grd =  meshgrid(linspace(-0.1, 1.1, 200),linspace(-0.1, 1.1, 200))   
	N_grd = length(x_grd)
	phix_grd_0  = Array{Float64,1}[[x_grd[i], y_grd[i]] for i=1:N_grd]
	
	phix_grd_1, Dphix_grd_1 = forward_flow(lambda_sigma[2], kappa, eta_coeff, phix_grd_0, Array{Float64,2}[eye(2) for i in 1:N_grd])
	det_grd = Float64[abs(det(Dphix_grd_1[i][1])) for i=1:N_grd]
	den, placeholder = target(phix_grd_0)
	est_den = det_grd .* den
	
	fig = plt.figure()
	plt.scatter(Float64[point[1] for point in X], Float64[point[2] for point in X], c="b")
	plt.contour(x_grd, y_grd, reshape(est_den,size(x_grd)), 30 )
	plt.savefig("out/example2_v$fignum.pdf",dpi=180)
	plt.close(fig)
end


#  gradient ascent on kappa and eta_coeff
lambda_sigma = [10.0, 0.05] 
for counter = 1:25
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,2}[eye(2) for i in 1:N])
	eta_coeff += prodc(0.002, dleta_coeff)
	toc()
end
saveim(1)


#  gradient ascent on kappa and eta_coeff
lambda_sigma = [5.0, 0.05] 
for counter = 1:25
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,2}[eye(2) for i in 1:N])
	eta_coeff += prodc(0.002, dleta_coeff)
	toc()
end
saveim(2)


#  gradient ascent on kappa and eta_coeff
lambda_sigma = [1.0, 0.05] 
for counter = 1:25
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,2}[eye(2) for i in 1:N])
	eta_coeff += prodc(0.002, dleta_coeff)
	toc()
end
saveim(3)

