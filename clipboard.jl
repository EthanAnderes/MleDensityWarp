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





function d_eta_dt(eta_coeff::Arrayd{1}, kappa::Arrayd{1})
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


