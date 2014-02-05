typealias Arrayd{oneOrTwo} Array{Array{Float64,oneOrTwo},1}


#---------------------------------------
# the flow eta, kappa, phix and Dphix forward
#---------------------------------------
function d_eta_dt!(dEtaDt::Arrayd{1},  eta_coeff::Arrayd{1}, kappa::Arrayd{1}, sigma, epsilon)
	for i = 1:length(dEtaDt)
		dEtaDt[i] = zero(kappa[1]) # you've got to start it at zero
		for j = 1:length(kappa) 
			dEtaDt[i] -=  epsilon * dot(eta_coeff[i],eta_coeff[j]) * gradR(kappa[i], kappa[j], sigma) 
		end
	end
end
function d_kappa_dt!(dKappaDt::Arrayd{1}, eta_coeff::Arrayd{1}, kappa::Arrayd{1}, sigma, epsilon)
	for i = 1:length(dKappaDt)
		dKappaDt[i] = zero(kappa[1]) # you've got to start it at zero
		for j = 1:length(kappa) 
			dKappaDt[i] += epsilon * eta_coeff[j] * R(kappa[i], kappa[j], sigma) 
		end
	end
end
function d_phix_dt!(dPhixDt::Arrayd{1}, eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, sigma, epsilon)
	for i = 1:length(dPhixDt)
		dPhixDt[i] = zero(phix[1])
		for j = 1:length(kappa) 
			dPhixDt[i] += epsilon * eta_coeff[j]* R(phix[i], kappa[j], sigma) 
		end
	end
end
function d_Dphix_dt!{oneOrTwo}(dDphixDt::Arrayd{oneOrTwo}, eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, Dphix::Arrayd{oneOrTwo}, sigma, epsilon)
	for i = 1:length(dDphixDt)
		dDphixDt[i] = zero(Dphix[1]) # you've got to start it at zero
		for j = 1:length(kappa) 
			dDphixDt[i] += epsilon * eta_coeff[j] * transpose(gradR(phix[i], kappa[j], sigma)) * Dphix[i]
		end
	end
end


#---------------------------------------
# the following are for the transpose flow
#---------------------------------------
# transDphix   = Array{Float64,2}[zeros(dim,dim) for i in 1:N]
function transd_dlDphix_dt!{oneOrTwo}(transDphix::Arrayd{oneOrTwo}, eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, dlDphix::Arrayd{oneOrTwo}, sigma, epsilon)
	dim = length(kappa[1])
	for i = 1:length(transDphix), col = 1:dim 
		transDphix[i][:,col] = zero(transDphix[1][:,col])
		for j = 1:length(eta_coeff) 
			transDphix[i][:,col] =  transDphix[i][:,col] + epsilon * gradR(phix[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlDphix[i][:,col]
		end
	end
end
# transdlphix   = Array{Float64,1}[zeros(dim) for i in 1:N]
function transd_dlphix_dt!{oneOrTwo}(transdlphix::Arrayd{1}, eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, dlphix::Arrayd{1}, Dphix::Arrayd{oneOrTwo}, dlDphix::Arrayd{oneOrTwo}, sigma, epsilon)
	dim = length(kappa[1])
	for i = 1:length(transdlphix)
		transdlphix[i] = zero(transdlphix[1])
		for j = 1:length(kappa)
			transdlphix[i] += epsilon * gradR(phix[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlphix[i]
			for col = 1:dim
				transdlphix[i] += epsilon * g1g1R(phix[i], kappa[j], sigma) * Dphix[i][:,col] * transpose(eta_coeff[j]) * dlDphix[i][:,col]
			end
		end
	end
end
# transdlKappa   = Array{Float64,1}[zeros(dim) for i in 1:N]
function transd_dlkappa_dt!{oneOrTwo}(transdlKappa::Arrayd{1}, eta_coeff::Arrayd{1}, dleta_coeff::Arrayd{1}, kappa::Arrayd{1}, dlkappa::Arrayd{1}, phix::Arrayd{1}, dlphix::Arrayd{1}, Dphix::Arrayd{oneOrTwo}, dlDphix::Arrayd{oneOrTwo}, sigma, epsilon)
	dim = length(kappa[1])
	for i = 1:length(transdlKappa)
		transdlKappa[i] = zero(transdlKappa[1])
		for j = 1:length(kappa)
			transdlKappa[i] -= epsilon * dot(eta_coeff[i],eta_coeff[j]) .* (g1g1R(kappa[i], kappa[j], sigma) * dleta_coeff[i])
			transdlKappa[i] -= epsilon * dot(eta_coeff[i],eta_coeff[j]) .* (g1g2R(kappa[j], kappa[i], sigma) * dleta_coeff[j])
			transdlKappa[i] += epsilon * gradR(kappa[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlkappa[i]
			transdlKappa[i] -= epsilon * gradR(kappa[j], kappa[i],sigma) * transpose(eta_coeff[i]) * dlkappa[j]
		end
	end
	for i = 1:length(transdlKappa)
		for j = 1:length(phix)
			transdlKappa[i] -= epsilon * gradR(phix[j], kappa[i],sigma) * transpose(eta_coeff[i]) * dlphix[j]
			for col = 1:dim
				transdlKappa[i] +=  epsilon * g1g2R(phix[j], kappa[i],sigma) * Dphix[j][:,col] * transpose(eta_coeff[i]) * dlDphix[j][:,col]
			end
		end
	end
end
# transdlEta   = Array{Float64,1}[zeros(dim) for i in 1:N]
function transd_dleta_dt!{oneOrTwo}(transdlEta::Arrayd{1}, eta_coeff::Arrayd{1}, dleta_coeff::Arrayd{1}, kappa::Arrayd{1}, dlkappa::Arrayd{1}, phix::Arrayd{1}, dlphix::Arrayd{1}, Dphix::Arrayd{oneOrTwo}, dlDphix::Arrayd{oneOrTwo}, sigma, epsilon)
	dim = length(kappa[1])
	for i = 1:length(kappa) 
		transdlEta[i] = zero(transdlEta[i])
		for j = 1:length(kappa)
			transdlEta[i] -=  epsilon * eta_coeff[j] * transpose(gradR(kappa[i], kappa[j],sigma)) * dleta_coeff[i]
			transdlEta[i] -=  epsilon * eta_coeff[j] * transpose(gradR(kappa[j], kappa[i],sigma)) * dleta_coeff[j]
			transdlEta[i] +=  epsilon * R(kappa[j], kappa[i], sigma) * dlkappa[j]
		end
	end
	for i = 1:length(kappa), j = 1:length(phix)
		transdlEta[i] += epsilon * R(phix[j], kappa[i], sigma) * dlphix[j]
		for col = 1:dim
			transdlEta[i] += epsilon * dot(Dphix[j][:,col], gradR(phix[j], kappa[i],sigma)) * dlDphix[j][:,col]
		end
	end
end




#---------------------------------------
# kernel evals and derivatives...these are local to this module
#------------------------------------
function r(d,sigma)
	 exp(-0.5 * d .* d ./ (sigma * sigma))
end
function rp(d,sigma)
	 -d .* r(d,sigma) ./ (sigma * sigma) 
end
# # test:  
# d = rand(3) 
# delta_d = 0.00001*[0,1,0]
# ( (r(d+delta_d, 3) - r(d, 3))/.00001  )[2] ==? rp(d,3)[2]
function rp_div_d(d,sigma) 
	 -r(d,sigma) ./ (sigma * sigma) 
end
function rpp(d,sigma)
	 d .* d .* r(d,sigma) ./ (sigma * sigma * sigma * sigma) - r(d,sigma) ./ (sigma * sigma)
end
function rppp(d,sigma)
	 -d .* d .* d .* r(d,sigma) ./ (sigma.^6.0) +  3.0 * d .* r(d,sigma) ./(sigma * sigma * sigma * sigma)
end
function R{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	r(norm(x-y),sigma)
end
function gradR{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	v=x-y
	n=norm(v)
	rp_div_d(n,sigma) .* v
end
function outer{T<:Number}(u::Array{T,1},v::Array{T,1})
	length(u) == 1 ? u[1]*v[1] : u*transpose(v)
end
function g1g2R{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	v = x-y
	n = norm(v)
	u = v/n
	eey = length(x) == 1 ? 1.0 : eye(length(x))
	if n == 0
		G = - rpp(n,sigma) * eey 
	else
		G = -rp_div_d(n,sigma) * eey
		G += outer(u,-u) * (rpp(n,sigma)-rp_div_d(n,sigma)) 
		return G
	end
end
function g1g1R{T<:Number}(x::Array{T,1},y::Array{T,1},sigma)
	 -1.0 * g1g2R(x,y,sigma)
end

#------------------------------------
# misc functions
#--------------------------------------------
function meshgrid(side_x,side_y)
	x = repmat(reshape([side_x],(1,length(side_x))) ,length(side_y),1)
	y = repmat(reshape([side_y],(length(side_y),1)) ,1,length(side_x))
	x,y
end
function prodc{oneOrTwo}(pp::Float64, cellA::Arrayd{oneOrTwo})
	Array{Float64,oneOrTwo}[ pp*cellA[i] for i = 1:length(cellA) ]
end


