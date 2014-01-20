#---------------------------------------
# the flow eta, kappa, phix and Dphix forward
#---------------------------------------
function d_dt(lambda_sigma, epsilon, eta_coeff, kappa, phix, Dphix)
	et_c = prodc(lambda_sigma[1] * epsilon,  d_eta_dt(eta_coeff, kappa, lambda_sigma[2]))
	ka   = prodc(lambda_sigma[1] * epsilon,  d_kappa_dt(eta_coeff, kappa, lambda_sigma[2]))
	ph   = prodc(lambda_sigma[1] * epsilon,  d_phix_dt(eta_coeff, kappa, phix, lambda_sigma[2]))
	Dph  = prodc(lambda_sigma[1] * epsilon,  d_Dphix_dt(eta_coeff, kappa, phix, Dphix, lambda_sigma[2]))
	et_c, ka, ph, Dph
end
function d_dt(lambda_sigma, epsilon, eta_coeff, kappa, phix, Dphix, phix_grd, Dphix_grd)
	et_c = prodc(lambda_sigma[1] * epsilon,  d_eta_dt(eta_coeff, kappa, lambda_sigma[2]))
	ka   = prodc(lambda_sigma[1] * epsilon,  d_kappa_dt(eta_coeff, kappa, lambda_sigma[2]))
	ph   = prodc(lambda_sigma[1] * epsilon,  d_phix_dt(eta_coeff, kappa, phix, lambda_sigma[2]))
	Dph  = prodc(lambda_sigma[1] * epsilon,  d_Dphix_dt(eta_coeff, kappa, phix, Dphix, lambda_sigma[2]))
    ph_g = prodc(lambda_sigma[1] * epsilon,  d_phix_dt(eta_coeff, kappa, phix_grd, lambda_sigma[2]))
    Dph_g= prodc(lambda_sigma[1] * epsilon,  d_Dphix_dt(eta_coeff, kappa, phix_grd, Dphix_grd, lambda_sigma[2]))
    et_c, ka, ph, Dph, ph_g, Dph_g
end
function d_eta_dt(eta_coeff, kappa, sigma)
	dd = size(eta_coeff[1])
	tmp = Array{Float64,1}[zeros(dd) for i in 1:length(eta_coeff)]
	for i = 1:length(kappa)
	 	for j = 1:length(kappa) 
			tmp[i] -= dot(eta_coeff[i],eta_coeff[j]) * gradR(kappa[i], kappa[j],sigma) 
		end
	end
	tmp
end
function d_kappa_dt(eta_coeff, kappa, sigma)
	dd = size(kappa[1])
	tmp = Array{Float64,1}[zeros(dd) for i in 1:length(kappa)]
	for i = 1:length(kappa)
	 	for j = 1:length(kappa) 
			tmp[i] += eta_coeff[j]* R(kappa[i], kappa[j],sigma) 
		end
	end
	tmp
end
function d_phix_dt(eta_coeff, kappa, phix, sigma)
	dd = size(phix[1])
	tmp = Array{Float64,1}[zeros(dd) for i in 1:length(phix)]
	for i = 1:length(phix)
	 	for j = 1:length(kappa) 
			tmp[i] += eta_coeff[j]* R(phix[i], kappa[j], sigma) 
		end
	end
	tmp
end
function d_Dphix_dt{dim}(eta_coeff, kappa, phix, Dphix::Array{Array{Float64,dim},1}, sigma)
	dd = size(Dphix[1])
	tmp = Array{Float64,dim}[zeros(dd) for i in 1:length(phix)]
	for i = 1:length(Dphix)
	 	for j = 1:length(kappa) 
			tmp[i] += eta_coeff[j] * (gradR(phix[i], kappa[j],sigma).') * Dphix[i]
		end
	end
	tmp
end



#---------------------------------------
# the following are for the transpose flow
#---------------------------------------
function transd_dt(lambda_sigma, epsilon, eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix)
	et_c, ka, ph, Dph = d_dt(lambda_sigma, epsilon, eta_coeff, kappa, phix, Dphix)
	det_c = prodc(lambda_sigma[1] * epsilon,  transd_dleta_dt(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, lambda_sigma[2]))
	dka   = prodc(lambda_sigma[1] * epsilon,  transd_dlkappa_dt(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, lambda_sigma[2]))
	dph   = prodc(lambda_sigma[1] * epsilon,  transd_dlphix_dt(eta_coeff,  kappa, phix, dlphix, Dphix, dlDphix, lambda_sigma[2]))
	dDph  = prodc(lambda_sigma[1] * epsilon,  transd_dlDphix_dt(eta_coeff, kappa, phix, dlDphix, lambda_sigma[2]))
	et_c, ka, ph, Dph, det_c, dka, dph, dDph
end
function transd_dlDphix_dt{dim}(eta_coeff,  kappa,  phix, dlDphix::Array{Array{Float64,dim},1}, sigma)
	n_knots = length(eta_coeff)
	n_phis  = length(dlDphix)
	returnval = dim == 1 ? Array{Float64,1}[[0.0] for i in 1:n_phis] : Array{Float64,dim}[zeros(dim,dim) for i in 1:n_phis]
	for col = 1:dim, i = 1:n_phis, j = 1:n_knots
		returnval[i][:,col] += gradR(phix[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlDphix[i][:,col]
	end
	prodc(-1.0, returnval)
end
function transd_dlphix_dt{dim}(eta_coeff, kappa, phix, dlphix, Dphix, dlDphix::Array{Array{Float64,dim},1}, sigma)
	n_knots  = length(eta_coeff)
	n_phis  = length(dlphix)
	returnval = Array{Float64,1}[zeros(dim) for i in 1:n_phis]
	for i = 1:n_phis, j = 1:n_knots
		returnval[i] +=  gradR(phix[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlphix[i]
		for col = 1:dim
			returnval[i] +=  g1g1R(phix[i], kappa[j],sigma) * Dphix[i][:,col] * transpose(eta_coeff[j]) * dlDphix[i][:,col]
		end
	end
	prodc(-1.0, returnval)
end
function transd_dlkappa_dt{dim}(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix::Array{Array{Float64,dim},1}, sigma)
	n_knots  = length(eta_coeff)
	n_phis   = length(dlphix)
	returnval = Array{Float64,1}[zeros(dim) for i in 1:n_knots]
	for i = 1:n_knots, j = 1:n_knots
		returnval[i] +=  -dot(eta_coeff[i],eta_coeff[j]) .* (g1g1R(kappa[i], kappa[j], sigma) * dleta_coeff[i])
		returnval[i] +=  -dot(eta_coeff[i],eta_coeff[j]) .* (g1g2R(kappa[j], kappa[i], sigma) * dleta_coeff[j])
		returnval[i] +=  -1.0 * gradR(kappa[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlkappa[i]
		returnval[i] +=         gradR(kappa[j], kappa[i],sigma) * transpose(eta_coeff[i]) * dlkappa[j]
	end
	for i = 1:n_knots, j = 1:n_phis
		returnval[i] +=  -1.0*gradR(phix[j], kappa[i],sigma) * transpose(eta_coeff[i]) * dlphix[j]
		for col = 1:dim
			returnval[i] +=  g1g2R(phix[j], kappa[i],sigma) * Dphix[j][:,col] * transpose(eta_coeff[i]) * dlDphix[j][:,col]
		end
	end
	prodc(-1.0, returnval)
end
function transd_dleta_dt{dim}(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix::Array{Array{Float64,dim},1}, sigma)
	n_knots = length(kappa)
	n_phis  = length(phix)
	returnval = Array{Float64,1}[ zeros(dim) for i in 1:n_knots]
	for i = 1:n_knots, j = 1:n_knots
		returnval[i] +=  -eta_coeff[j] * transpose(gradR(kappa[i], kappa[j],sigma)) * dleta_coeff[i]
		returnval[i] +=  -eta_coeff[j] * transpose(gradR(kappa[j], kappa[i],sigma)) * dleta_coeff[j]
		returnval[i] +=  R(kappa[j], kappa[i], sigma) * dlkappa[j]
	end
	for i = 1:n_knots, j = 1:n_phis
		returnval[i] +=  R(phix[j], kappa[i], sigma) * dlphix[j]
		for col = 1:dim
			returnval[i] += dot(Dphix[j][:,col], gradR(phix[j], kappa[i],sigma)) * dlDphix[j][:,col]
		end
	end
	prodc(-1.0, returnval)
end



#------------------------------------
# misc function
#------------------------------------
function prodc{n}(pp::Float64, cellA::Array{Array{Float64,n},1})
	Array{Float64,n}[ pp*cellA[i] for i = 1:length(cellA) ]
end



#---------------------------------------
# kernel evals and derivatives...these are local to this module
#------------------------------------
function r(d,sigma)
	 exp(-.5*(d/sigma).^2)
end
function rp(d,sigma)
	 -d ./ (sigma^2 * exp(d.^2./(2*sigma^2)))
end
# # test:  
# d = rand(dim) 
# delta_d = 0.00001*[0,1,0]
# ( (r(d+delta_d, 3) - r(d, 3))/.00001  )[2] ==? rp(d,3)[2]
function rp_div_d(d,sigma) 
	 -1.0./(sigma^2*exp(d.^2/(2*sigma^2)))
end
function rpp(d,sigma)
	 d.^2./(sigma^4*exp(d.^2/(2*sigma^2))) - 1./(sigma^2*exp(d.^2/(2*sigma^2)))
end
"TODO: test everything below this line"
function rppp(d,sigma)
	 (3*d)./(sigma^4*exp(d.^2/(2*sigma^2))) - d.^3./(sigma^6*exp(d.^2/(2*sigma^2)))
end
function R{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	r(norm(x-y),sigma)
end
function gradR{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	v=x-y
	n=norm(v)
	rp_div_d(n,sigma).*v
end
function outer{T<:Number}(u::Array{T,1},v::Array{T,1})
	length(u) == 1 ? u[1]*v[1] : u*transpose(v)
end
function g1g2R{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	v=x-y
	n=norm(v)
	eey = length(x) == 1 ? 1.0 : eye(length(x))
	G=rp_div_d(n,sigma)*eey
	if n!=0
		u=v/n
		G+=outer(u,-u) *(rpp(n,sigma)-rp_div_d(n,sigma))
	end
	G
end
function g1g1R{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	 -1*g1g2R(x,y,sigma)
end



#---------------------------------------
# for testing some of the above code
#------------------------------------
# # test the functions in the module using the following inputs
# dim = 2 # dimension
# sigma = 2.0       # the scale of the reproducing kernel
# n_phis  = 20
# n_knots = 10
# phix      = {rand(dim) for i in 1:n_phis }
# Dphix     = {rand(dim,dim) for i in 1:n_phis}
# eta_coeff = {rand(dim) for i in 1:n_knots }
# kappa     = {rand(dim) for i in 1:n_knots }

# dlphix      = {rand(dim) for i in 1:n_phis }
# dlDphix     = {rand(dim,dim) for i in 1:n_phis}
# dleta_coeff = {rand(dim) for i in 1:n_knots }
# dlkappa     = {rand(dim) for i in 1:n_knots }




#---------------------------------------
# forward in time evolution of delta eta, etc. 
# I don't think I need these. Only the transpose flow
#---------------------------------------
# function d_dleta_dt(eta_coeff, dleta_coeff, kappa, dlkappa, sigma)
# 	d1Rmat   = {gradR(ki, kj,sigma) for ki in kappa, kj in kappa}
# 	d1d1Rmat = {g1g1R(ki, kj,sigma) for ki in kappa, kj in kappa}
# 	d1d2Rmat = {g1g2R(ki, kj,sigma) for ki in kappa, kj in kappa}
# 	dim = length(kappa[1])
# 	n_knots  = length(kappa)
# 	returnval = {zeros(dim) for i in 1:n_knots}
# 	for i = 1:n_knots, j = 1:n_knots
# 		returnval[i] += d1Rmat[i,j] * transpose(-eta_coeff[j]) * dleta_coeff[i]
# 		returnval[i] += d1Rmat[i,j] * transpose(-eta_coeff[i]) * dleta_coeff[j]
# 		returnval[i] += -1*dot(eta_coeff[i],eta_coeff[j]) * d1d1Rmat[i,j] * dlkappa[i]
# 		returnval[i] += -1*dot(eta_coeff[i],eta_coeff[j]) * d1d2Rmat[i,j] * dlkappa[j]
# 	end
# 	returnval
# end
# function d_dlkappa_dt(eta_coeff, dleta_coeff, kappa, dlkappa, sigma)
# 	Rmat     = {R(ki, kj,sigma) for ki in kappa, kj in kappa}
# 	d1Rmat   = {gradR(ki, kj,sigma) for ki in kappa, kj in kappa}
# 	d2Rmat   = {-gradR(ki, kj,sigma) for ki in kappa, kj in kappa}
# 	dim = length(kappa[1])
# 	n_knots  = length(kappa)
# 	returnval = {zeros(dim) for i in 1:n_knots}
# 	for i = 1:n_knots, j = 1:n_knots
# 		returnval[i] += Rmat[i,j] * dleta_coeff[j]
# 		returnval[i] += eta_coeff[j] * d1Rmat[i,j] * dlkappa[i]
# 		returnval[i] += eta_coeff[j] * d2Rmat[i,j] * dlkappa[j]
# 	end
# 	returnval
# end
# function d_dlphix_dt(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, sigma)
# 	Rmat     = {     R(ph, kj,sigma) for ph in phix, kj in kappa}
# 	d1Rmat   = { gradR(ph, kj,sigma) for ph in phix, kj in kappa}
# 	d2Rmat   = {-gradR(ph, kj,sigma) for ph in phix, kj in kappa}
# 	dim = length(kappa[1])
# 	n_phis  = length(phix)
# 	n_knots  = length(kappa)
# 	returnval = {zeros(dim) for i in 1:n_phis}
# 	for i = 1:n_phis, j = 1:n_knots
# 		returnval[i] += Rmat[i,j] * dleta_coeff[j]
# 		returnval[i] += eta_coeff[j] * dot(d1Rmat[i,j], dlphix[i])
# 		returnval[i] += eta_coeff[j] * dot(d2Rmat[i,j], dlkappa[j])
# 	end
# 	returnval
# end
# function d_dlDphix_dt(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma)
# 	d1Rmat   = { gradR(ph, kj,sigma) for ph in phix, kj in kappa}
# 	d1d1Rmat = { g1g1R(ph, kj,sigma) for ph in phix, kj in kappa}
# 	d1d2Rmat = { g1g2R(ph, kj,sigma) for ph in phix, kj in kappa}
# 	dim = length(kappa[1])
# 	n_phis  = length(phix)
# 	n_knots  = length(kappa)
# 	returnval = {zeros(dim,dim) for i in 1:n_phis}
# 	for col = 1:dim, i = 1:n_phis, j = 1:n_knots
# 		returnval[i][:,col] += dot(d1Rmat[i,j], Dphix[i][:,col]) * dleta_coeff[j]
# 		returnval[i][:,col] += eta_coeff[j] * (dot(dlDphix[i][:,col], d1d1Rmat[i,j] * dlphix[i]) )
# 		returnval[i][:,col] += eta_coeff[j] * (dot(dlDphix[i][:,col], d1d2Rmat[i,j] * dlkappa[j]) )
# 		returnval[i][:,col] += eta_coeff[j] * dot(d1Rmat[i,j], dlDphix[i][:,col])
# 	end
# 	returnval
# end

