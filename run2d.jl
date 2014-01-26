include("src/ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")

# set some algorithm parameters
lambda_sigma = [0.1, 0.1]  #lambda  = smoothness penalty coeff and sigma =  the scale of the reproducing kernel

# generate the data:  X
tmpx = [rand(55), randn(50)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)
X = Array{Float64,1}[[tmpx[i]-mean(tmpx), tmpy[i]-mean(tmpy)] for i=1:N]

# initialize kappa and eta_coeff
kappa_new     = Array{Float64,1}[X[i]  for i in 1:N ]
kappa_init = deepcopy(kappa_new)
#kappa     = Array{Float64,1}[]
#x_grd_kns, y_grd_kns =  meshgrid(linspace(-1.0,1.0, 25),linspace(-1.0,1.0, 25))
#append!(kappa, Array{Float64,1}[ [x_grd_kns[i], y_grd_kns[i]] for i in 1:length(x_grd_kns)])
n_knots   = length(kappa_new)
eta_coeff_new = Array{Float64,1}[zero(kappa_new[i]) for i in 1:n_knots ]

# initialize phix and Dphix which are used to compute the likelihood
phix_new      = copy(X)
Dphix_new     = Array{Float64,2}[eye(2) for i in 1:N]

# initialze the points which are used to visualize the density
x_grd, y_grd =  meshgrid(linspace(-1.0,1.0, 150),linspace(-1.0,1.0, 150))   
N_grd = length(x_grd)
phix_grd_0  = Array{Float64,1}[[x_grd[i], y_grd[i]] for i=1:N_grd]

# set the target measure
target = hfun32d # this sets an alias to a function giving in the file targets.jl


#-------------------------------------------------------
#  gradient ascent on kappa and eta_coeff
#--------------------------------------------------------
for counter = 1:40
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, copy(kappa_new), copy(eta_coeff_new), copy(phix_new), copy(Dphix_new))
	# kappa     += prodc(0.01, dlkappa)
	eta_coeff_new += prodc(0.005, dleta_coeff)
	toc()
end


#----------------------------------------------------------
#  visualize
#----------------------------------------------------------
using  PyCall
@pyimport matplotlib.pyplot as plt 
phix_grd_1, Dphix_grd_1 = forward_flow(lambda_sigma[2], copy(kappa_new), copy(eta_coeff_new), copy(phix_grd_0), Array{Float64,2}[eye(2) for i in 1:N_grd])
det_grd = Float64[abs(det(Dphix_grd_1[i][1])) for i=1:N_grd]
den, placeholder = target(phix_grd_0)
est_den = det_grd .* den
#fig = plt.figure(figsize=(12,6))
#plt.subplot(1,2,1)
	plt.scatter(Float64[point[1] for point in X], Float64[point[2] for point in X], c="b")
	plt.contour(x_grd, y_grd, reshape(est_den,size(x_grd)),30 )
	#plt.subplot(1,2,2)
#	plt.scatter(Float64[point[1] for point in kappa_new], Float64[point[2] for point in kappa_new], c="r")
#	plt.scatter(Float64[point[1] for point in kappa_init], Float64[point[2] for point in kappa_init], c="b")
plt.show()



# savepath = "images/run1"
# if isdir(savepath) 
# 	run(`rm -r $savepath`)
# end
# run(`mkdir $savepath`)
# plt.savefig("$savepath/density.png",dpi=180)
# plt.close(fig)


