include("src/ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")

# set some algorithm parameters
lambda_sigma = [2.0, 0.05]  #lambda  = smoothness penalty coeff and sigma =  the scale of the reproducing kernel

# generate the data:  X
tmpx = [rand(50), randn(70)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)

X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx), (tmpy[i]-minimum(tmpy))/maximum(tmpy)] for i=1:N]
# X = Array{Float64,1}[[(tmpx[i]-mean(tmpx))/std(tmpx), (tmpy[i]-mean(tmpy))/std(tmpy)] for i=1:N]

# target(x) = targetNormal2d(x) # this sets an alias to a function giving in the file targets.jl
target(x) = targetUnif2d(x; targSig = 0.1, limit = 0.25, center = 0.5) # this sets an alias to a function giving in the file targets.jl


# initialize kappa and eta_coeff
kappa     = Array{Float64,1}[X[i]  for i in 1:round(N/2)]
# x_grd_kns, y_grd_kns =  meshgrid(linspace(-1.0,1.0, 10),linspace(-1.0,1.0, 10))
# append!(kappa, Array{Float64,1}[ [x_grd_kns[i], y_grd_kns[i]] for i in 1:length(x_grd_kns)])
n_knots   = length(kappa)
eta_coeff = Array{Float64,1}[zero(kappa[i]) for i in 1:n_knots]

#-------------------------------------------------------
#  gradient ascent on kappa and eta_coeff
#-------------------------------------------------------
for counter = 1:25
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,2}[eye(2) for i in 1:N])
	# kappa     += prodc(0.01, dlkappa)
	eta_coeff += prodc(0.01, dleta_coeff)
	toc()
end


#----------------------------------------------------------
#  visualize
#----------------------------------------------------------
x_grd, y_grd =  meshgrid(linspace(-0.1, 1.1, 200),linspace(-0.1, 1.1, 200))   
N_grd = length(x_grd)
phix_grd_0  = Array{Float64,1}[[x_grd[i], y_grd[i]] for i=1:N_grd]

using  PyCall
@pyimport matplotlib.pyplot as plt 
phix_grd_1, Dphix_grd_1 = forward_flow(lambda_sigma[2], kappa, eta_coeff, phix_grd_0, Array{Float64,2}[eye(2) for i in 1:N_grd])
det_grd = Float64[abs(det(Dphix_grd_1[i][1])) for i=1:N_grd]
den, placeholder = target(phix_grd_0)
est_den = det_grd .* den

fig = plt.figure()
# plt.subplot(1,2,1)
	plt.scatter(Float64[point[1] for point in X], Float64[point[2] for point in X], c="b")
	plt.contour(x_grd, y_grd, reshape(est_den,size(x_grd)), 30 )
# plt.subplot(1,2,2)
#	plt.scatter(Float64[point[1] for point in kappa], Float64[point[2] for point in kappa], c="r")
#	plt.scatter(Float64[point[1] for point in kappa_init], Float64[point[2] for point in kappa_init], c="b")
plt.savefig("out/run2dv2.pdf",dpi=180)
plt.close(fig)
