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
include("src/flow_ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")


# typealias Arrayd{dim} Array{Array{Float64,dim},1}
tmpx = [rand(50), randn(75)/10.0 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)
X = Array{Float64,1}[[tmpx[i]-mean(tmpx), tmpy[i]-mean(tmpy)] for i=1:N]
kappa_cell     = Array{Float64,1}[X[i]       for i in 1:N ]
eta_coeff_cell = Array{Float64,1}[zero(kappa_cell[i]) for i in 1:N ]

kappa_splat     = Array(Float64,(2,N))
eta_coeff_splat     = Array(Float64,(2,N))
for i = 1:N
	kappa_splat[:,i] = X[i] 
	eta_coeff_splat[:,i] = [0.0, 0.0]
end
function d_eta_dt1(eta_coeff::Array{Float64,2}, kappa::Array{Float64,2})
	sigma = 1.0
	dd = size(kappa,1)
	tmp = zeros(kappa)
	for i = 1:size(kappa,2)
	 	for j = 1:size(kappa,2)
	 		dtij = dot(eta_coeff[:,i],eta_coeff[:,j]) 
	 		gRij = gradR(kappa[:,i], kappa[:,j],sigma)
	 		for z = 1:size(kappa,1)
				tmp[z,i] -=  dtij*gRij[z]
			end
		end
	end
	tmp
end
function d_eta_dt2(eta_coeff::Array{Float64,2}, kappa::Array{Float64,2})
	sigma = 1.0
	dd = size(kappa,1)
	tmp = zeros(kappa)
	for i = 1:size(kappa,2)
	 	for j = 1:size(kappa,2)
	 		dtij = 0.0
	 		for z = 1:size(kappa,1)
	 			dtij += eta_coeff[z,i]*eta_coeff[z,j]
	 		end
	 		gRij = gradR([1.0, 2.0], [1.0, 2.0],sigma)
	 		for z = 1:size(kappa,1)
				tmp[z,i] -=  dtij*gRij[z]
			end
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


d_eta_dt1(eta_coeff_splat, kappa_splat)
d_eta_dt2(eta_coeff_splat, kappa_splat)
d_eta_dt(eta_coeff_cell, kappa_cell)


@time d_eta_dt1(eta_coeff_splat, kappa_splat);
@time d_eta_dt2(eta_coeff_splat, kappa_splat);
@time d_eta_dt(eta_coeff_cell, kappa_cell);






#########################################


# speed check for Array{Float64,2} vrs Array{Array{Float64,1},1}
using  PyCall
@pyimport matplotlib.pyplot as plt 
include("src/flow_ode.jl")
include("src/rfuncs.jl")
include("src/targets.jl")
# typealias
# typealias Arrayd{dim} Array{Array{Float64,dim},1}
tmpx = [rand(50), randn(75)/10.0 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)
X = Array{Float64,1}[[tmpx[i]-mean(tmpx), tmpy[i]-mean(tmpy)] for i=1:N]
kappa_cell     = Array{Float64,1}[X[i]       for i in 1:N ]
eta_coeff_cell = Array{Float64,1}[zero(kappa_cell[i]) for i in 1:N ]


    


function d_eta_dt(eta_coeff::Arrayd{1}, kappa::Arrayd{1}, sigma, epsilon)
	dEtaDt   = Array{Float64,1}[zeros(2) for i in 1:length(eta_coeff)]
	for i = 1:length(dEtaDt)
		dEtaDt[i] = zero(kappa[1]) # you've got to start it at zero
		for j = 1:length(kappa) 
			dEtaDt[i] -=  epsilon * dot(eta_coeff[i],eta_coeff[j]) * gradR(kappa[i], kappa[j], sigma) 
		end
	end
	dEtaDt
end
function d_kappa_dt(eta_coeff::Arrayd{1}, kappa::Arrayd{1}, sigma, epsilon)
	dKappaDt = Array{Float64,1}[zeros(2) for i in 1:length(kappa)]
	for i = 1:length(dKappaDt)
		dKappaDt[i] = zero(kappa[1]) # you've got to start it at zero
		for j = 1:length(kappa) 
			dKappaDt[i] += epsilon * eta_coeff[j] * R(kappa[i], kappa[j], sigma) 
		end
	end
	dKappaDt
end
function d_phix_dt(eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, sigma, epsilon)
	dPhixDt  = Array{Float64,1}[zeros(2) for i in 1:length(phix)]
	for i = 1:length(dPhixDt)
		dPhixDt[i] = zero(phix[1])
		for j = 1:length(kappa) 
			dPhixDt[i] += epsilon * eta_coeff[j]* R(phix[i], kappa[j], sigma) 
		end
	end
	dPhixDt
end
function d_Dphix_dt(eta_coeff::Arrayd{1}, kappa::Arrayd{1}, phix::Arrayd{1}, Dphix::Arrayd{2}, sigma, epsilon)
    dDphixDt = Array{Float64,2}[zeros(2,2) for i in 1:length(phix)]
	for i = 1:length(dDphixDt)
		dDphixDt[i] = zero(Dphix[1]) # you've got to start it at zero
		for j = 1:length(kappa) 
			dDphixDt[i] += epsilon * eta_coeff[j] * transpose(gradR(phix[i], kappa[j], sigma)) * Dphix[i]
		end
	end
	dDphixDt
end

sigma = 0.1
epsilon = 0.1
D =  Array{Float64,2}[ones(2,2) for i in 1:length(X)]
@time d_eta_dt(eta_coeff_cell, kappa_cell, sigma, epsilon);
@time d_kappa_dt(eta_coeff_cell, kappa_cell, sigma, epsilon);
@time d_phix_dt(eta_coeff_cell, kappa_cell, X, sigma, epsilon);
@time d_Dphix_dt(eta_coeff_cell, kappa_cell, X, D, sigma, epsilon);
@time td_Dphix_dt(eta_coeff_cell, kappa_cell, X, D, sigma, epsilon);









# from the  above code it is clear that the bottle neck is gradR not the looping



type Flow
	kappa::Array{Float64,2}
	eta_coeff::Array{Float64,2}
end

flowinstance1 = Flow(rand(2,2), rand(3,3))
flowinstance1.kappa # prints out kappa
flowinstance1.eta_coeff # prints out eta

# if you want a default for one of the parameters you can make these constructors
Flow(x)= Flow(x, eye(2,2))
Flow()= Flow(eye(2,2), eye(2,2))

flowinstance2 = Flow(rand(2,2))
flowinstance3 = Flow()

# Define + on Flow types
function +(a::Flow, b::Flow) 
	Flow(a.kappa + b.kappa, a.eta_coeff + b.eta_coeff)
end

flowinstance2 + flowinstance3

# to try and make things faster you can define an immutable type which tells the complier that 
# the fields will not be re-bound.


immutable ImFlow
	kappa::Array{Float64,2}
	eta_coeff::Array{Float64,2}
end

imflowinstance1 = ImFlow(rand(2,2), rand(3,3))

imflowinstance1.kappa = eye(2,2) # this gives an error since I'm trying to re-bind kappa
imflowinstance1.kappa[1] = 2.0 # this is ok since I'm just changing the entry kappa[1] and not the container kappa


##################

#---------------------------------------
# kernel evals and derivatives...these are local to this module
#------------------------------------
r(d::Real,sigma) = exp(-0.5 * d * d / (sigma * sigma))
function rp_div_d(d::Real,sigma) 
	s2 = sigma * sigma
	-exp(-0.5 * d * d / s2) / s2
end
function rpp(d::Real,sigma)
	rd = r(d,sigma)
	s2 = sigma * sigma
	d * d * rd / (s2 * s2) - rd / s2
end
function R{T<:Real}(x::Array{T,1},y::Array{T,1},sigma)
	r(norm(x-y),sigma)
end
function gradR{T<:Real}(x::Array{T,1},y::Array{T,1},sigma)
	v=x-y
	n=norm(v)
	v * rp_div_d(n,sigma)
end
function outer{T<:Real}(u::Array{T,1},v::Array{T,1})
	length(u) == 1 ? u[1]*v[1] : u*transpose(v)
end
function g1g2R{T<:Real}(x::Array{T,1},y::Array{T,1},sigma)
	v = x-y
	n = norm(v)
	u = v/n
	eey = length(x) == 1 ? 1.0 : eye(length(x))
	rpd = rp_div_d(n,sigma)
	if n == 0
		G = -rpp(n,sigma) * eey 
		return G
	else
		G = -rpd * eey
		G += outer(u,-u) * (rpp(n,sigma) - rpd) 
		return G
	end
end
function g1g1R{T<:Real}(x::Array{T,1},y::Array{T,1},sigma)
	 -g1g2R(x,y,sigma)
end



x=[1.0, 2.0]
y=[1.0, 2.2]
@time g1g2R(x, y, 2.0)
@time g1g2R2(x, y, 2.0)

@time g1g2R(x, y, 2.0)
@time g1g2R2(x, y, 2.0)




#---------------------
# I want to test the impact of  the type conversion
# ------------------------

include("src/rfuncs.jl")
typealias Array1 Array{Array{Float64,1},1} # for lists of vectors
typealias Array2 Array{Array{Float64,2},1} # for lists of jacobians
array1(dim, k) = Array{Float64,1}[zeros(dim) for i=1:k] # constructor for Array1
array2(dim, k) = Array{Float64,2}[zeros(dim, dim) for i=1:k] # constructor for Array2
array2eye(dim, k) = Array{Float64,2}[eye(dim) for i=1:k] # constructor for Array2
immutable Flow
	kappa::Array1
	eta_coeff::Array1
	phix::Array1
	Dphix::Array2
	dim::Int64	
end
Flow(dim::Int64, k) = Flow(array1(dim, k), array1(dim, k), array1(dim, k), array2(dim, k), dim) # zero Flow constructor
for opt = (:+, :-, :*) # this defines +, - and * between Flow and Float64
	@eval begin
		function ($opt)(yin::Flow, a::Real)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = Flow(yin.dim, 0)
			for i = 1:nKap
				push!(yout.kappa, ($opt)(a,yin.kappa[i])) 
				push!(yout.eta_coeff, ($opt)(a,yin.eta_coeff[i])) 
			end
			for i = 1:nPhi
				push!(yout.phix, ($opt)(a,yin.phix[i]))
				push!(yout.Dphix, ($opt)(a,yin.Dphix[i])) 
			end
			yout
		end # end function definition
	end # end the begin block
	@eval ($opt)(a::Real, yin::Flow) = ($opt)(yin::Flow, a::Float64)
end # end for loop over functions
# ytest1 = Flow(2, 3)
# ytest2 = ytest1 + 5.7
# ytest3 = ytest2 * 1.7 + 4
function d_forward_dt(yin::Flow, sigma)
	nKap = length(yin.kappa)
	nPhi = length(yin.phix)
	yout = Flow(yin.dim, nKap)
	# eta
	for i = 1:nKap, j = 1:nKap 
		yout.eta_coeff[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) * gradR(yin.kappa[i], yin.kappa[j], sigma) 
	end
	# kappa
	for i = 1:nKap, j = 1:nKap
		yout.kappa[i] += yin.eta_coeff[j] * R(yin.kappa[i], yin.kappa[j], sigma) 
	end
	# phi
	for i = 1:nPhi, j = 1:nKap 
		yout.phix[i] += yin.eta_coeff[j] * R(yin.phix[i], yin.kappa[j], sigma) 
	end
	# Dphi
	for i = 1:nPhi, j = 1:nKap 
		yout.Dphix[i] += yin.eta_coeff[j] * transpose(gradR(yin.phix[i], yin.kappa[j], sigma)) * yin.Dphix[i]
	end
	yout
end

#transpose
immutable TrFlow
	kappa::Array1
	eta_coeff::Array1
	phix::Array1
	Dphix::Array2
	dlkappa::Array1
	dleta_coeff::Array1
	dlphix::Array1
	dlDphix::Array2
	dim::Int64	
end

TrFlow(dim::Int64, k) = TrFlow(array1(dim, k), array1(dim, k), array1(dim, k), array2(dim, k), array1(dim, k), array1(dim, k), array1(dim, k), array2(dim, k), dim) # zero Flow constructor

for opt = (:+, :-, :*) # this defines +, - and * between Flow and Float64
	@eval begin
		function ($opt)(yin::TrFlow, a::Real)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = TrFlow(yin.dim, 0)
			for i = 1:nKap
				push!(yout.kappa, ($opt)(a,yin.kappa[i])) 
				push!(yout.eta_coeff, ($opt)(a,yin.eta_coeff[i])) 
				push!(yout.dlkappa, ($opt)(a,yin.dlkappa[i])) 
				push!(yout.dleta_coeff, ($opt)(a,yin.dleta_coeff[i])) 
			end
			for i = 1:nPhi
				push!(yout.phix, ($opt)(a,yin.phix[i]))
				push!(yout.Dphix, ($opt)(a,yin.Dphix[i])) 
				push!(yout.dlphix, ($opt)(a,yin.dlphix[i]))
				push!(yout.dlDphix, ($opt)(a,yin.dlDphix[i])) 
			end
			yout
		end # end function definition
	end # end the begin block
	@eval ($opt)(a::Real, yin::Flow) = ($opt)(yin::Flow, a::Float64)
end # end for loop over functions

function d_backward_dt(yin::TrFlow, sigma)
	nKap = length(yin.kappa)
	nPhi = length(yin.phix)
	yout = TrFlow(yin.dim, nKap)
	# eta
	for i = 1:nKap, j = 1:nKap 
		yout.eta_coeff[i] += dot(yin.eta_coeff[i], yin.eta_coeff[j]) * gradR(yin.kappa[i], yin.kappa[j], sigma) 
	end
	# kappa
	for i = 1:nKap, j = 1:nKap
		yout.kappa[i] -= yin.eta_coeff[j] * R(yin.kappa[i], yin.kappa[j], sigma) 
	end
	# phi
	for i = 1:nPhi, j = 1:nKap 
		yout.phix[i] -= yin.eta_coeff[j] * R(yin.phix[i], yin.kappa[j], sigma) 
	end
	# Dphi
	for i = 1:nPhi, j = 1:nKap 
		yout.Dphix[i] -= yin.eta_coeff[j] * transpose(gradR(yin.phix[i], yin.kappa[j], sigma)) * yin.Dphix[i]
	end
	# dlDphix
	for col = 1:yin.dim, i = 1:nPhi, j = 1:nKap
		yout.dlDphix[i][:,col] +=  gradR(yin.phix[i], yin.kappa[j], sigma) * transpose(yin.eta_coeff[j]) * yin.dlDphix[i][:,col]
	end
	# dlphix
	for i = 1:nPhi, j = 1:nKap
		yout.dlphix[i] += gradR(yin.phix[i], yin.kappa[j],sigma) * transpose(yin.eta_coeff[j]) * yin.dlphix[i]
		for col = 1:yin.dim
			yout.dlphix[i] +=  g1g1R(yin.phix[i], yin.kappa[j], sigma) * yin.Dphix[i][:,col] * transpose(yin.eta_coeff[j]) * yin.dlDphix[i][:,col]
		end
	end
	# dlkappa
	for i = 1:nKap, j = 1:nKap
		yout.dlkappa[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) .* (g1g1R(yin.kappa[i], yin.kappa[j], sigma) * yin.dleta_coeff[i])
		yout.dlkappa[i] -= dot(yin.eta_coeff[i], yin.eta_coeff[j]) .* (g1g2R(yin.kappa[j], yin.kappa[i], sigma) * yin.dleta_coeff[j])
		yout.dlkappa[i] += gradR(yin.kappa[i], yin.kappa[j],sigma) * transpose(yin.eta_coeff[j]) * yin.dlkappa[i]
		yout.dlkappa[i] -= gradR(yin.kappa[j], yin.kappa[i],sigma) * transpose(yin.eta_coeff[i]) * yin.dlkappa[j]
	end
	for i = 1:nKap, j = 1:nPhi
		yout.dlkappa[i] -= gradR(yin.phix[j], yin.kappa[i], sigma) * transpose(yin.eta_coeff[i]) * yin.dlphix[j]
		for col = 1:yin.dim
			yout.dlkappa[i] += g1g2R(yin.phix[j], yin.kappa[i],sigma) * yin.Dphix[j][:,col] * transpose(yin.eta_coeff[i]) * yin.dlDphix[j][:,col]
		end
	end 
	# dleta_coeff
	for i = 1:nKap, j = 1:nKap
		yout.dleta_coeff[i] -= yin.eta_coeff[j] * transpose(gradR(yin.kappa[i], yin.kappa[j], sigma)) * yin.dleta_coeff[i]
		yout.dleta_coeff[i] -= yin.eta_coeff[j] * transpose(gradR(yin.kappa[j], yin.kappa[i], sigma)) * yin.dleta_coeff[j]
		yout.dleta_coeff[i] += R(yin.kappa[j], yin.kappa[i], sigma) * yin.dlkappa[j]
	end
	for i = 1:nKap, j = 1:nPhi
		yout.dleta_coeff[i] += R(yin.phix[j], yin.kappa[i], sigma) * yin.dlphix[j]
		for col = 1:yin.dim
			yout.dleta_coeff[i] += dot(yin.Dphix[j][:,col], gradR(yin.phix[j], yin.kappa[i],sigma)) * yin.dlDphix[j][:,col]
		end
	end
	yout
end




tmpx = [rand(50), randn(75)/10.0 + .8]; tmpy = tmpx + rand(size(tmpx)) .* 1.1; nKap = length(tmpx)
X = Array{Float64,1}[[tmpx[i]-mean(tmpx), tmpy[i]-mean(tmpy)] for i=1:nKap]

dim = length(X[1])
yFlow = Flow(deepcopy(X), array1(dim,nKap), deepcopy(X), array2(dim,nKap), dim)
yTrFlow = TrFlow(deepcopy(X), array1(dim,nKap), deepcopy(X), array2(dim,nKap), deepcopy(X), array1(dim,nKap), deepcopy(X), array2(dim,nKap), dim)


@time d_forward_dt(yFlow, 1.0);
@time d_backward_dt(yTrFlow, 1.0);


