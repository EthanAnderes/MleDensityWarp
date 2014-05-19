# 2 dimensional example which produces a 3 by 3 table of densities, one for each pair of (lambda, kernel scale)
push!(LOAD_PATH, homedir()*"/Box Sync/MleDensityWarp/mlewarp_repo/src")
require("flowType.jl")
require("grad.jl")
require("targets.jl")
using  PyCall
@pyimport matplotlib.pyplot as plt 

#---- set the data and the target
dim = 2
tmpx = shuffle([rand(50), randn(80)/10 .+ .8])
tmpy = tmpx + rand(size(tmpx)) .* 1.1
nPhi = length(tmpx)
X = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx), (tmpy[i]-minimum(tmpy))/maximum(tmpy)] for i=1:nPhi]
# target(x) = targetUnif2d(x; targSig = 0.1, limit = 0.4, center = 0.5)
target(x) = targetNormal2d(x; targSig = 0.5, center = [0.5,0.5])
using GaussKernel # sets which kernel you use

#---- generate Flow object
kappa     = Array{Float64,1}[X[i]  for i in 1:round(nPhi/2)]
# x_grd_kns, y_grd_kns =  meshgrid(linspace(-1.0,1.0, 10),linspace(-1.0,1.0, 10))
# append!(kappa, Array{Float64,1}[ [x_grd_kns[i], y_grd_kns[i]] for i in 1:length(x_grd_kns)])
nKap   = length(kappa)
y0 = Flow(kappa, array1(dim, nKap), X, array2eye(dim, nPhi), dim) 

# set an array of lambda, sigmas
arrayLambda = (10.0, 1.0, 0.1)
arraySigma = (10.0, 1.0, 0.1)

#  gradient ascent on kappa and eta_coeff
ycontainer = Array{Flow,2}
est_den_container = Array{Array{Float64,2},2}
for lcntr = 1:3, scntr = 1:3
	for counter = 1:50
		tic()
		z0 = get_grad(y0, arrayLambda[lcntr], arraySigma[scntr])
		y0 = y0 + 0.002 * z0
		toc()
	end
	ycontainer[lcntr, scntr] = deepcopy(y0)
	#-- now we save the estimated density
	x_grd, y_grd =  meshgrid(linspace(-0.1, 1.1, 175),linspace(-0.1, 1.1, 175))   
	N_grd = length(x_grd)
	phix_grd_0  = Array{Float64,1}[[x_grd[i], y_grd[i]] for i=1:N_grd]
	Dphix_grd_0 = Array{Float64,2}[eye(2) for i in 1:N_grd]
	yplt0 = Flow(y0.kappa, y0.eta_coeff, phix_grd_0, Dphix_grd_0, dim) 
    dydt(t,y)= d_forward_dt(y, sigma)
    (t1,yplt1)=ode23_abstract_end(dydt,[0,1], yplt0) # Flow y0 forward to time 1
	det_grd = Float64[abs(yplt1.Dphix[i][1]) for i=1:N_grd]
	den, placeholder = target(phix_grd_0)
	est_den_container[lcntr, scntr] = det_grd.*den
end




#--- function to save images
function saveim(fignum)
	fig = plt.figure()
	plt.scatter(Float64[point[1] for point in X], Float64[point[2] for point in X], c="b")
	plt.contour(x_grd, y_grd, reshape(est_den,size(x_grd)), 30 )
	plt.savefig("out/example2_v$fignum.pdf",dpi=180)
	plt.close(fig)
end

