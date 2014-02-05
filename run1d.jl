include("src/ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")

# set some algorithm parameters
lambda_sigma = [0.3, 0.05]  #lambda  = smoothness penalty coeff and sigma =  the scale of the reproducing kernel

# generate the data:  X and set the target measure
tmpx = [rand(45), randn(75)/10 + .8]
N = length(tmpx)

# X = Array{Float64,1}[[(tmpx[i]-mean(tmpx))/std(tmpx)] for i=1:N]
X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx)] for i=1:N]

# target(x) = targetNormal1d(x) # this sets an alias to a function giving in the file targets.jl
target(x) = targetUnif1d(x; targSig = 0.1, limit = 0.35, center = 0.5) # this sets an alias to a function giving in the file targets.jl

# initialize kappa and eta_coeff
kappa     = Array{Float64,1}[X[i]  for i in 1:round(N)]
# append!(kappa, Array{Float64,1}[[linspace(-0.1,1.1, 25)[i]]  for i in 1:25])
n_knots   = length(kappa)
eta_coeff = Array{Float64,1}[zero(kappa[i]) for i in 1:n_knots]

#-------------------------------------------------------
#  gradient ascent on kappa and eta_coeff
#--------------------------------------------------------
for counter = 1:35
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, X, Array{Float64,1}[[1.0] for i in 1:N])
	# kappa     += prodc(0.01, dlkappa)
	eta_coeff += prodc(0.005, dleta_coeff)
	toc()
end


#----------------------------------------------------------
#  visualize
#----------------------------------------------------------

# initialze the points which are used to visualize the density
x_grd = linspace(-0.2,1.2, 350)  
N_grd = length(x_grd)
phix_grd_0  = Array{Float64,1}[[x_grd[i]] for i=1:N_grd]

using  PyCall
@pyimport matplotlib.pyplot as plt 

phix_grd_1, Dphix_grd_1 = forward_flow(lambda_sigma[2], kappa, eta_coeff, phix_grd_0, Array{Float64,1}[[1.0] for i in 1:N_grd])
det_grd = Float64[abs(Dphix_grd_1[i][1]) for i=1:N_grd]
den, placeholder = target(phix_grd_0)
est_den = det_grd.*den

fig = plt.figure()
# plt.stem(tmpx, tmpx.*0 + mean(est_den), "-.")
plt.hist([pnt[1] for pnt in X], 50, normed=1, histtype="stepfilled")
plt.plot(x_grd, est_den)
plt.savefig("out/run1dv1.pdf",dpi=180)
plt.close(fig)

