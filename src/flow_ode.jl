#---------------------------------
#   Helper function to compute the forward d/dt of kappa,eta,phix,Dphix
#--------------------------------

function d_kepDp_dt(t,y,sigma,dim,Nkappa,Nphix,size_Dphix)
    # input y is assumed to be of the form {kappa,eta_coeff,phix,Dphix}
    # First, unpack this
    (kappa,eta_coeff,phix,Dphix)=y
    kappa=convert(Array{Array{Float64,1},1},kappa)
    eta_coeff=convert(Array{Array{Float64,1},1},eta_coeff)
    phix=convert(Array{Array{Float64,1},1},phix)
    Dphix=convert(Array{Array{Float64,length(size_Dphix)},1},Dphix)

    # Then construct new structures to hold the d/dt's
    dKappaDt = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
    dEtaDt   = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
    dPhixDt  = Array{Float64,1}[zeros(dim) for i in 1:Nphix]
    dDphixDt = Array{Float64,length(size_Dphix)}[zeros(size_Dphix) for i in 1:Nphix]

    # And compute their new values (by mutation)
    d_kappa_dt!(dKappaDt, eta_coeff, kappa, sigma, 1.0)
    d_eta_dt!(dEtaDt,     eta_coeff, kappa, sigma, 1.0)
    d_phix_dt!(dPhixDt,   eta_coeff, kappa, phix, sigma, 1.0)
    d_Dphix_dt!(dDphixDt, eta_coeff, kappa, phix, Dphix, sigma, 1.0)
    # Finally return the d/dt in the same order as y
    {dKappaDt,dEtaDt,dPhixDt,dDphixDt}
end

include("ode_tool.jl")
using ODE_TOOL

#---------------------------------
#   get the gradient
#--------------------------------
# kappa is geting mutated
function get_grad!{oneOrtwo}(lambda_sigma, kappa, eta_coeff, phix, Dphix::Array{Array{Float64,oneOrtwo},1})
	stepsODE = 100
	epsilon = 1.0/stepsODE
	Nkappa = length(kappa)
	Nphix = length(phix)
	sigma = lambda_sigma[2]
	dim = length(phix[1])
	size_Dphix=size(Dphix[1])

	y0={kappa,eta_coeff,phix,Dphix}
	dydt(t,y)=d_kepDp_dt(t,y,sigma,dim,Nkappa,Nphix,size(Dphix[1]))
	# Flow y0 forward to time 1
	(t1,y1)=ode23_abstract_end(dydt,[0,1],y0)
	#Update using the time 1 state
	(kappa,eta_coeff,phix,Dphix)=y1
	#Restore types
	kappa=convert(Array{Array{Float64,1},1},kappa)
	eta_coeff=convert(Array{Array{Float64,1},1},eta_coeff)
	phix=convert(Array{Array{Float64,1},1},phix)
	Dphix=convert(Array{Array{Float64,length(size_Dphix)},1},Dphix)

	## initialize the delta 
	#dEtaDt   = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
	#dKappaDt = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
	#dPhixDt  = Array{Float64,1}[zeros(dim) for i in 1:Nphix]
	#dDphixDt = Array{Float64,oneOrtwo}[zero(Dphix[1]) for i in 1:Nphix]
	## time = 0 now
	#for counter = 1:stepsODE # this flows forward
	#	d_eta_dt!(dEtaDt,     eta_coeff, kappa, sigma, epsilon)
	#	d_kappa_dt!(dKappaDt, eta_coeff, kappa, sigma, epsilon)
	#	d_phix_dt!(dPhixDt,   eta_coeff, kappa, phix, sigma, epsilon)
	#	d_Dphix_dt!(dDphixDt, eta_coeff, kappa, phix, Dphix, sigma, epsilon)
	#	for k = 1:Nkappa
	#		eta_coeff[k] += dEtaDt[k]
	#		kappa[k]     += dKappaDt[k]
	#	end
	#	for k = 1:Nphix
	#		phix[k]      += dPhixDt[k]
	#		Dphix[k]     += dDphixDt[k]
	#	end
	#end
	## time = 1 now

	dKappaDt = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
	dEtaDt   = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
	dPhixDt  = Array{Float64,1}[zeros(dim) for i in 1:Nphix]
	dDphixDt = Array{Float64,length(size_Dphix)}[zeros(size_Dphix) for i in 1:Nphix]

	dlphix      = Array{Float64,1}[target(phix)[2][i]/Nphix for i = 1:Nphix]
	if oneOrtwo == 1
		dlDphix = Array{Float64,1}[(1/Dphix[i])/Nphix for i = 1:Nphix]
	else
		dlDphix = Array{Float64,2}[inv(Dphix[i])/Nphix for i = 1:Nphix]
	end
	dleta_coeff = Array{Float64,1}[zeros(dim) for i = 1:Nkappa]
	dlkappa     = Array{Float64,1}[zeros(dim) for i = 1:Nkappa]
	transDphix   = Array{Float64,oneOrtwo}[zero(Dphix[1]) for i in 1:Nphix]
	transdlphix   = Array{Float64,1}[zeros(dim) for i in 1:Nphix]
	transdlKappa   = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
	transdlEta   = Array{Float64,1}[zeros(dim) for i in 1:Nkappa]
	for counter = 1:stepsODE #this flows backwards
		d_eta_dt!(dEtaDt,  eta_coeff, kappa, sigma, epsilon)
		d_kappa_dt!(dKappaDt, eta_coeff, kappa, sigma, epsilon)
		d_phix_dt!(dPhixDt, eta_coeff, kappa, phix, sigma, epsilon)
		d_Dphix_dt!(dDphixDt, eta_coeff, kappa, phix, Dphix, sigma, epsilon)
		transd_dlDphix_dt!(transDphix, eta_coeff, kappa, phix, dlDphix, sigma, epsilon)
		transd_dlphix_dt!(transdlphix, eta_coeff, kappa, phix, dlphix, Dphix, dlDphix, sigma, epsilon)
		transd_dlkappa_dt!(transdlKappa, eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma, epsilon)
		transd_dleta_dt!(transdlEta, eta_coeff, dleta_coeff, kappa, dlkappa, phix, dlphix, Dphix, dlDphix, sigma, epsilon)
		for k = 1:Nkappa
			eta_coeff[k]  -= dEtaDt[k]
			kappa[k]      -= dKappaDt[k]
			dleta_coeff[k]+= transdlEta[k]
			dlkappa[k]    += transdlKappa[k]
		end
		for k = 1:Nphix
			phix[k]       -= dPhixDt[k]
			Dphix[k]      -= dDphixDt[k]
			dlphix[k]     += transdlphix[k]
			dlDphix[k]    += transDphix[k]
		end
	end
	# time = 0 now
	# add the regularization
	"need to update kappa too"
	for i = 1:length(eta_coeff)
		for j = 1:length(kappa) 
			dleta_coeff[i] -= lambda_sigma[1] * eta_coeff[j] * R(kappa[i], kappa[j], sigma) 
		end
	end
	dlkappa, dleta_coeff
end

# this is the function all which doesn't mutate the arguments
get_grad(lambda_sigma, kappa, eta_coeff, phix, Dphix) = get_grad!(lambda_sigma, deepcopy(kappa), deepcopy(eta_coeff), deepcopy(phix), deepcopy(Dphix)) 

#---------------------------------
#   flow forward
#---------------------------------
function forward_flow!{oneOrtwo}(sigma, kappa, eta_coeff, phix_grd, Dphix_grd::Array{Array{Float64,oneOrtwo},1})
	stepsODE = 100
	epsilon = 1.0/stepsODE
	Nkappa = length(kappa)
	Nphix = length(phix_grd)
	# initialize the delta s
	dEtaDt   = Array{Float64,1}[zero(kappa[1]) for i in 1:Nkappa]
	dKappaDt = Array{Float64,1}[zero(kappa[1]) for i in 1:Nkappa]
	dPhixDt_grd  = Array{Float64,1}[zero(kappa[1]) for i in 1:Nphix]
	dDphixDt_grd = Array{Float64,oneOrtwo}[zero(Dphix_grd[1]) for i in 1:Nphix]
	for counter = 1:stepsODE
		d_eta_dt!(dEtaDt,  eta_coeff, kappa, sigma, epsilon)
		d_kappa_dt!(dKappaDt, eta_coeff, kappa, sigma, epsilon)
		d_phix_dt!(dPhixDt_grd, eta_coeff, kappa, phix_grd, sigma, epsilon)
		d_Dphix_dt!(dDphixDt_grd, eta_coeff, kappa, phix_grd, Dphix_grd, sigma, epsilon)
		for k = 1:Nkappa
			eta_coeff[k] +=  dEtaDt[k]
			kappa[k]    +=  dKappaDt[k]
		end
		for k = 1:Nphix
			phix_grd[k]   +=  dPhixDt_grd[k]
			Dphix_grd[k]  +=  dDphixDt_grd[k]
		end
	end
	phix_grd, Dphix_grd
end

forward_flow(sigma, kappa, eta_coeff, phix_grd, Dphix_grd) = forward_flow!(sigma, deepcopy(kappa), deepcopy(eta_coeff), deepcopy(phix_grd), deepcopy(Dphix_grd))






function sqez(X)
	Xmat = X[1]
	for k=2:length(X)
		Xmat = [Xmat X[k]]
	end
	Xmat
end





# plt.plot(sqez(X)[1,:], sqez(X)[2,:],"."); plt.show()
