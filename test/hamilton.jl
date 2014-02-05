# test to make sure our geodeics have constant hamiltonians over time
include("../src/ode.jl")
include("../src/rfuncs.jl")
include("../src/targets.jl")

tmpx = [rand(50), randn(70)/10 + .8]
tmpy = tmpx + rand(size(tmpx)) .* 1.1
N = length(tmpx)

kappa = Array{Float64,1}[[(tmpx[i]- minimum(tmpx))/maximum(tmpx), (tmpy[i]-minimum(tmpy))/maximum(tmpy)] for i=1:N]
eta_coeff = Array{Float64,1}[ randn(size(kappa[i]))/100.0 for i in 1:length(kappa)]

sigma = 0.1

function henergy(kappa, eta_coeff, sigma)
	sum = 0.0
	N = length(kappa)
	for i=1:N, j=1:N
		sum += dot(eta_coeff[i], eta_coeff[j]) * R(kappa[i], kappa[j], sigma)
	end
	sum
end
function forward_test!(ham, kappa, eta_coeff, sigma)
	# this function records the time varying hamiltonian and pushes the values to ham
	stepsODE = 1000
	epsilon = 1.0/stepsODE
	Nkappa = length(kappa)
	# initialize the delta s
	dEtaDt   = Array{Float64,1}[zero(kappa[1]) for i in 1:Nkappa]
	dKappaDt = Array{Float64,1}[zero(kappa[1]) for i in 1:Nkappa]
	for counter = 1:stepsODE
		d_eta_dt!(dEtaDt,  eta_coeff, kappa, sigma, epsilon)
		d_kappa_dt!(dKappaDt, eta_coeff, kappa, sigma, epsilon)
		for k = 1:Nkappa
			eta_coeff[k] +=  dEtaDt[k]
			kappa[k]    +=  dKappaDt[k]
		end
		push!(ham, henergy(kappa, eta_coeff, sigma))
	end
end



ham = Float64[henergy(kappa, eta_coeff, sigma)]
forward_test!(ham, kappa, eta_coeff, sigma)

using PyCall	
@pyimport matplotlib.pyplot as plt
plt.plot(push!(ham,0.0))
plt.show()