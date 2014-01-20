#---------------------------------
#   get the gradient
#--------------------------------
function get_grad{dim}(lambda_sigma, kappa, eta_coeff, phix, Dphix::Array{Array{Float64,dim},1})
	stepsODE = 50
	epsilon = 1.0/stepsODE
	# time = 0 now
	for counter = 1:stepsODE #this flows forward
		d1, d2, d3, d4  =  d_dt(lambda_sigma, epsilon, eta_coeff, kappa, phix, Dphix)
		eta_coeff += d1
		kappa     += d2
		phix      += d3
		Dphix     += d4 
	end
	# time = 1 now
	dlphix      = Array{Float64,1}[target(phix)[2][i]/length(phix) for i = 1:length(phix)]
	if dim == 1
		dlDphix = Array{Float64,dim}[(1/Dphix[i])/length(phix) for i = 1:length(phix)]
	else
		dlDphix = Array{Float64,dim}[inv(Dphix[i])/length(phix) for i = 1:length(phix)]
	end
	dleta_coeff = Array{Float64,1}[zeros(dim) for i = 1:length(kappa)]
	dlkappa     = Array{Float64,1}[zeros(dim) for i = 1:length(kappa)]
	for counter = 1:stepsODE #this flows backwards
		d1, d2, d3, d4, dd1, dd2, dd3, dd4  =  transd_dt(lambda_sigma, epsilon, eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix)
		eta_coeff 	-= d1
		kappa     	-= d2
		phix      	-= d3
		Dphix     	-= d4 
		dleta_coeff -= dd1
		dlkappa     -= dd2
		dlphix      -= dd3
		dlDphix     -= dd4 
	end
	# time = 0 now
	dlkappa, dleta_coeff
end


#---------------------------------
#   flow forward
#---------------------------------
function forward_flow(lambda_sigma, kappa, eta_coeff, phix, Dphix, phix_grd, Dphix_grd)
	stepsODE = 50
	epsilon = 1.0/stepsODE
	for counter = 1:stepsODE
		d1, d2, d3, d4, ph_g, Dph_g  =  d_dt(lambda_sigma, epsilon, eta_coeff, kappa, phix, Dphix, phix_grd, Dphix_grd)
		eta_coeff += d1
		kappa     += d2
		phix      += d3
		Dphix     += d4   
		phix_grd  += ph_g
		Dphix_grd += Dph_g
	end
	phix_grd, Dphix_grd
end
function sqez(X)
	Xmat = X[1]
	for k=2:length(X)
		Xmat = [Xmat X[k]]
	end
	Xmat
end
# plt.plot(sqez(X)[1,:], sqez(X)[2,:],"."); plt.show()
