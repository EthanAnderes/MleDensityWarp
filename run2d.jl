using  PyCall
@pyimport matplotlib.pyplot as plt 
include("src/ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")

# set some algorithm parameters
dim     = 2 # dimension
lambda_sigma = [0.2, 0.1]  #lambda  = smoothness penalty coeff and sigma =  the scale of the reproducing kernel

# generate the data:  X
tmpx = [rand(50), randn(75)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 0.1
N = length(tmpx)
X = Array{Float64,1}[[tmpx[i], tmpy[i]] for i=1:N]

# initialize kappa and eta_coeff
n_knots   = round(1.0 * N)
kappa     = Array{Float64,1}[X[i]       for i in 1:n_knots ]
eta_coeff = Array{Float64,1}[zero(X[i]) for i in 1:n_knots ]
phix      = X
Dphix     = Array{Float64,2}[eye(2) for i in 1:n_knots]


function meshgrid(side_x,side_y)
	x = repmat(reshape([side_x],(1,length(side_x))) ,length(side_y),1)
	y = repmat(reshape([side_y],(length(side_y),1)) ,1,length(side_x))
	x,y
end
x_grd, y_grd =  meshgrid(linspace(-1.0,1.5, 50),linspace(-1.0,1.5, 50))   
N_grd = length(x_grd)
phix_grd_0  = Array{Float64,1}[[x_grd[i], y_grd[i]] for i=1:N_grd]

# set the target measure
target = targetUnif2d # this sets an alias to a function giving in the file targets.jl


#-------------------------------------------------------
#  gradient ascent on kappa and eta_coeff
#--------------------------------------------------------
for counter = 1:30
	tic()
	dlkappa, dleta_coeff = get_grad(lambda_sigma, kappa, eta_coeff, phix, Dphix)
	# kappa  += 0.01 * dlkappa
	eta_coeff += 0.001 * dleta_coeff
	toc()
end


#----------------------------------------------------------
#  visualize
#----------------------------------------------------------
fig = plt.figure()
phix_grd_1, Dphix_grd_1 = forward_flow(lambda_sigma, kappa, eta_coeff, X, Array{Float64,2}[eye(2) for i in 1:N], phix_grd_0, Array{Float64,2}[eye(2) for i in 1:N_grd])
det_grd = Float64[abs(Dphix_grd_1[i][1]) for i=1:N_grd]
den, placeholder = target(phix_grd_0)
est_den = det_grd.*den
# plt.stem(tmpx, tmpx.*0 + mean(est_den), "-.")
plt.hist(tmpx, 20, normed=1, histtype="stepfilled")
plt.plot(x_grd, est_den)
plt.show()

# savepath = "images/run1"
# if isdir(savepath) 
# 	run(`rm -r $savepath`)
# end
# run(`mkdir $savepath`)
# plt.savefig("$savepath/density.png",dpi=180)
# plt.close(fig)


