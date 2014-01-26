typealias Arrayd{dim} Array{Array{Float64,dim},1}

# generate the data:  X
tmpx = [rand(25), randn(50)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)
X = Array{Float64,1}[[tmpx[i]-mean(tmpx), tmpy[i]-mean(tmpy)] for i=1:N]
kappa     = Array{Float64,1}[X[i]       for i in 1:N ]
eta_coeff = deepcopy(kappa)
phix      = X
Dphix     = Array{Float64,2}[eye(2) for i in 1:N]
sigma = 0.1
epsilon = 0.02
dim = 2
#---------------------------------------
function r(d,sigma)
	 exp(-.5*(d/sigma).^2)
end
function rp(d,sigma)
	 -d ./ (sigma^2 * exp(d.^2./(2*sigma^2)))
end
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
function prodc{dim}(pp::Float64, cellA::Arrayd{dim})
	Array{Float64,dim}[ pp*cellA[i] for i = 1:length(cellA) ]
end
function target(kappa::Array{Array{Float64,1},1})
	targsig = 0.5
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k = 1:length(kappa)
		push!(density, exp(-(kappa[k][1].^2 + kappa[k][2].^2) / (2 * targsig^2)  ) / (2 * pi * targsig^2) )
		push!(gradH,[-kappa[k][1]/(targsig^2) , -kappa[k][2] / (targsig^2)]) 
	end
	density, gradH
end


#### test these ##########
function transd_dlDphix_dt{dim}(eta_coeff,  kappa,  phix, dlDphix::Arrayd{dim}, sigma)
	n_knots = length(eta_coeff)
	n_phis  = length(dlDphix)
	returnval = dim == 1 ? Array{Float64,1}[[0.0] for i in 1:n_phis] : Array{Float64,dim}[zeros(dim,dim) for i in 1:n_phis]
	for col = 1:dim, i = 1:n_phis, j = 1:n_knots
		returnval[i][:,col] += gradR(phix[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlDphix[i][:,col]
	end
	prodc(-1.0, returnval)
end
function transd_dlphix_dt{dim}(eta_coeff, kappa, phix, dlphix, Dphix, dlDphix::Arrayd{dim}, sigma)
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
function transd_dlkappa_dt{dim}(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix::Arrayd{dim}, sigma)
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
function transd_dleta_dt{dim}(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix::Arrayd{dim}, sigma)
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
#### against these ...test these ##########
# transDphix   = Array{Float64,2}[zeros(dim,dim) for i in 1:N]
function transd_dlDphix_dt!{dim}(transDphix::Arrayd{dim}, eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, dlDphix::Arrayd{dim}, sigma, epsilon)
	for i = 1:length(transDphix)
		transDphix[i] = zero(transDphix[1])
		for j = 1:length(eta_coeff) 
			for col = 1:dim 
				transDphix[i][:,col] =  transDphix[i][:,col] - epsilon * gradR(phix[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlDphix[i][:,col]
			end
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
				transdlphix[i] += epsilon * g1g1R(phix[i], kappa[j],sigma) * Dphix[i][:,col] * transpose(eta_coeff[j]) * dlDphix[i][:,col]
			end
		end
	end
end
# transdlKappa   = Array{Float64,1}[zeros(dim) for i in 1:N]
function transd_dlkappa_dt!{dim}(transdlKappa::Arrayd{1}, eta_coeff::Arrayd{1}, dleta_coeff::Arrayd{1}, kappa::Arrayd{1}, dlkappa::Arrayd{1}, phix::Arrayd{1}, dlphix::Arrayd{1}, Dphix::Arrayd{dim}, dlDphix::Arrayd{dim}, sigma, epsilon)
	for i = 1:length(transdlKappa)
		transdlKappa[1] = zero(transdlKappa[1])
		for j = 1:length(kappa)
			transdlKappa[i] += epsilon * dot(eta_coeff[i],eta_coeff[j]) .* (g1g1R(kappa[i], kappa[j], sigma) * dleta_coeff[i])
			transdlKappa[i] += epsilon * dot(eta_coeff[i],eta_coeff[j]) .* (g1g2R(kappa[j], kappa[i], sigma) * dleta_coeff[j])
			transdlKappa[i] += epsilon * gradR(kappa[i], kappa[j],sigma) * transpose(eta_coeff[j]) * dlkappa[i]
			transdlKappa[i] -= epsilon * gradR(kappa[j], kappa[i],sigma) * transpose(eta_coeff[i]) * dlkappa[j]
		end
	end
	for i = 1:length(transdlKappa)
		for j = 1:length(phix)
			transdlKappa[i] += epsilon * gradR(phix[j], kappa[i],sigma) * transpose(eta_coeff[i]) * dlphix[j]
			for col = 1:dim
				transdlKappa[i] -=  epsilon * g1g2R(phix[j], kappa[i],sigma) * Dphix[j][:,col] * transpose(eta_coeff[i]) * dlDphix[j][:,col]
			end
		end
	end
end
# transdlEta   = Array{Float64,1}[zeros(dim) for i in 1:N]
function transd_dleta_dt!{dim}(transdlEta::Arrayd{1}, eta_coeff::Arrayd{1}, dleta_coeff::Arrayd{1}, kappa::Arrayd{1}, dlkappa::Arrayd{1}, phix::Arrayd{1}, dlphix::Arrayd{1}, Dphix::Arrayd{dim}, dlDphix::Arrayd{dim}, sigma, epsilon)
	for i = 1:length(kappa) 
		transdlEta[i] = zero(transdlEta[i])
		for j = 1:length(kappa)
			transdlEta[i] +=  epsilon * eta_coeff[j] * transpose(gradR(kappa[i], kappa[j],sigma)) * dleta_coeff[i]
			transdlEta[i] +=  epsilon * eta_coeff[j] * transpose(gradR(kappa[j], kappa[i],sigma)) * dleta_coeff[j]
			transdlEta[i] -=  epsilon * R(kappa[j], kappa[i], sigma) * dlkappa[j]
		end
	end
	for i = 1:length(kappa), j = 1:length(phix)
		transdlEta[i] -= epsilon * R(phix[j], kappa[i], sigma) * dlphix[j]
		for col = 1:dim
			transdlEta[i] -= epsilon * dot(Dphix[j][:,col], gradR(phix[j], kappa[i],sigma)) * dlDphix[j][:,col]
		end
	end
end

dlphix      = Array{Float64,1}[target(phix)[2][i]/length(phix) for i = 1:length(phix)]
dleta_coeff = Array{Float64,1}[zeros(dim) for i = 1:N]
dlkappa     = Array{Float64,1}[zeros(dim) for i = 1:N]
transdlphix  = Array{Float64,1}[zeros(dim) for i in 1:N]
transdlKappa = Array{Float64,1}[zeros(dim) for i in 1:N]
transdlEta   = Array{Float64,1}[zeros(dim) for i in 1:N]
transDphix   = Array{Float64,2}[zeros(dim,dim) for i in 1:N]
dlDphix = Array{Float64,2}[inv(Dphix[i])/length(phix) for i = 1:length(phix)]

@time transDphixtest = transd_dlDphix_dt(eta_coeff,  kappa,  phix, dlDphix, sigma)
@time transd_dlDphix_dt!(transDphix, eta_coeff, kappa, phix, dlDphix, sigma, epsilon)

@time transdlphixtest = transd_dlphix_dt(eta_coeff, kappa, phix, dlphix, Dphix, dlDphix, sigma)
@time transd_dlphix_dt!(transdlphix, eta_coeff, kappa, phix, dlphix, Dphix, dlDphix, sigma, epsilon)

@time transdlKappatest = transd_dlkappa_dt(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma)
@time transd_dlkappa_dt!(transdlKappa, eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma, epsilon)

@time transdlEtatest = transd_dleta_dt(eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma)
@time transd_dleta_dt!(transdlEta, eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma, epsilon)

error("exiting the script now")

[transDphix prodc(epsilon,transDphixtest)] # same
[transdlphix prodc(epsilon, transdlphixtest)] # different
[transdlKappa prodc(epsilon, transdlKappatest)] #same
[transdlEta prodc(epsilon, transdlEtatest)]



######################################################################
######################################################################
######################################################################

# speed check for Array{Float64,2} vrs Array{Array{Float64,1},1}
using  PyCall
@pyimport matplotlib.pyplot as plt 
include("src/ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")


typealias Arrayd{dim} Array{Array{Float64,dim},1}
tmpx = [rand(50), randn(75)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)
X = Array{Float64,1}[[tmpx[i]-mean(tmpx), tmpy[i]-mean(tmpy)] for i=1:N]
kappa_cell     = Array{Float64,1}[X[i]       for i in 1:N ]
eta_coeff_cell = Array{Float64,1}[zero(kappa_cell[i]) for i in 1:N ]

# kappa_splat     = Array(Float64,(N,2))
# eta_coeff_splat     = Array(Float64,(N,2))
# for i = 1:N
# 	kappa_splat[i,:] = transpose(X[i]) 
# 	eta_coeff_splat[i,:] = [0.0 0.0]
# end
# function d_eta_dt(eta_coeff::Array{Float64,2}, kappa::Array{Float64,2})
# 	sigma = 1.0
# 	dd = size(kappa,1)
# 	tmp = zeros(kappa)
# 	for i = 1:size(kappa,2)
# 	 	for j = 1:size(kappa,2) 
# 			tmp[i,:] -= transpose(dot(eta_coeff[i,:][:],eta_coeff[i,:][:]) * gradR(kappa[i,:][:], kappa[i,:][:],sigma) )
# 		end
# 	end
# 	tmp
# end




kappa_splat     = Array(Float64,(2,N))
eta_coeff_splat     = Array(Float64,(2,N))
for i = 1:N
	kappa_splat[:,i] = X[i] 
	eta_coeff_splat[:,i] = [0.0, 0.0]
end
function d_eta_dt(eta_coeff::Array{Float64,2}, kappa::Array{Float64,2})
	sigma = 1.0
	dd = size(kappa,1)
	tmp = zeros(kappa)
	for i = 1:size(kappa,2)
	 	for j = 1:size(kappa,2) 
			tmp[:,i] = tmp[:,i] - dot(eta_coeff[:,i],eta_coeff[:,i]) * gradR(kappa[:,i], kappa[:,i],sigma)
		end
	end
	tmp
end





function d_eta_dt(eta_coeff, kappa)
	sigma = 1.0
	dd = size(eta_coeff[1])
	tmp = Array{Float64,1}[zeros(dd) for i in 1:length(eta_coeff)]
	for i = 1:length(kappa)
	 	for j = 1:length(kappa) 
			tmp[i] -= dot(eta_coeff[i],eta_coeff[j]) * gradR(kappa[i], kappa[j],sigma) 
		end
	end
	tmp
end


d_eta_dt(eta_coeff_splat, kappa_splat)
d_eta_dt(eta_coeff_cell, kappa_cell)



@time d_eta_dt(eta_coeff_cell, kappa_cell);
@time d_eta_dt(eta_coeff_splat, kappa_splat);


