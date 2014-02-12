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

# zero Flow constructor
Flow(dim, nKap, nPhi) = Flow(array1(dim, nKap), array1(dim, nKap), array1(dim, nPhi), array2(dim, nPhi), dim) 

# take a traspose flow, extract the dl's and convert to Flow (used to update the gradiet)
dFlow(z::TrFlow) = Flow(z.dlkappa, z.dleta_coeff, z.dlphix, z.dlDphix, z.dim)

# zero TrFlow constructor
TrFlow(dim, nKap, nPhi) = TrFlow(array1(dim, nKap), array1(dim, nKap), array1(dim, nPhi), array2(dim, nPhi), array1(dim, nKap), array1(dim, nKap), array1(dim, nPhi), array2(dim, nPhi), dim) # zero Flow constructor

# this is used to initialize the transpose flow at time 1 of a grad computation 
function TrFlow(yin::Flow, target::Function)
	nKap = length(yin.kappa)
	nPhi = length(yin.phix)
	dleta_coeff = array1(yin.dim, nKap)
	dlkappa     = array1(yin.dim, nKap)
    dlphix      = Array{Float64,1}[target(yin.phix)[2][i]/nPhi for i = 1:nPhi]
    dlDphix     = Array{Float64,2}[inv(yin.Dphix[i])/nPhi for i = 1:nPhi]
	TrFlow(yin.kappa, yin.eta_coeff, yin.phix, yin.Dphix, dlkappa, dleta_coeff, dlphix, dlDphix, yin.dim)
end

# defines + between Flow and TrFlow... used to update with  the return value of get_grad
function +(yin::Flow, zin::TrFlow) 
	nKap = length(yin.kappa) # I don't check that zin has same dims
	nPhi = length(yin.phix)
	yout = Flow(yin.dim, 0, 0) # return a Flow
	for i = 1:nKap
		push!(yout.kappa,     yin.kappa[i]     + zin.dlkappa[i]) 
		push!(yout.eta_coeff, yin.eta_coeff[i] + zin.dleta_coeff[i]) 
	end
	# we don't need to update these from TrFlow
	for i = 1:nPhi
		push!(yout.phix,   yin.phix[i])
		push!(yout.Dphix,  yin.Dphix[i]) 
	end
	yout
end 

# defines + between Flow and Flow, used in ode
function +(yin::Flow, zin::Flow) 
	nKap = length(yin.kappa) 
	nPhi = length(yin.phix)
	yout = Flow(yin.dim, 0, 0)
	for i = 1:nKap
		push!(yout.kappa,     yin.kappa[i]     + zin.kappa[i]) 
		push!(yout.eta_coeff, yin.eta_coeff[i] + zin.eta_coeff[i]) 
	end
	for i = 1:nPhi
		push!(yout.phix,  zin.phix[i] + yin.phix[i])
		push!(yout.Dphix, zin.Dphix[i]+ yin.Dphix[i]) 
	end
	yout
end 

# defines - between Flow and Flow, used in ode
function -(yin::Flow, zin::Flow) 
	nKap = length(yin.kappa) 
	nPhi = length(yin.phix)
	yout = Flow(yin.dim, 0, 0)
	for i = 1:nKap
		push!(yout.kappa,     yin.kappa[i]     - zin.kappa[i]) 
		push!(yout.eta_coeff, yin.eta_coeff[i] - zin.eta_coeff[i]) 
	end
	for i = 1:nPhi
		push!(yout.phix,  zin.phix[i] - yin.phix[i])
		push!(yout.Dphix, zin.Dphix[i]- yin.Dphix[i]) 
	end
	yout
end 

# defines + between TrFlow and TrFlow, used in ode
function +(yin::TrFlow, zin::TrFlow) 
	nKap = length(yin.kappa) 
	nPhi = length(yin.phix)
	yout = TrFlow(yin.dim, 0, 0)
	for i = 1:nKap
		push!(yout.kappa,     yin.kappa[i]     + zin.kappa[i]) 
		push!(yout.eta_coeff, yin.eta_coeff[i] + zin.eta_coeff[i]) 
		push!(yout.dlkappa,     yin.dlkappa[i]     + zin.dlkappa[i]) 
		push!(yout.dleta_coeff, yin.dleta_coeff[i] + zin.dleta_coeff[i]) 
	end
	for i = 1:nPhi
		push!(yout.phix,  zin.phix[i] + yin.phix[i])
		push!(yout.Dphix, zin.Dphix[i]+ yin.Dphix[i]) 
		push!(yout.dlphix,  zin.dlphix[i] + yin.dlphix[i])
		push!(yout.dlDphix, zin.dlDphix[i]+ yin.dlDphix[i]) 
	end
	yout
end 

# defines - between TrFlow and TrFlow, used in ode
function -(yin::TrFlow, zin::TrFlow) 
	nKap = length(yin.kappa) 
	nPhi = length(yin.phix)
	yout = TrFlow(yin.dim, 0, 0)
	for i = 1:nKap
		push!(yout.kappa,     yin.kappa[i]     - zin.kappa[i]) 
		push!(yout.eta_coeff, yin.eta_coeff[i] - zin.eta_coeff[i]) 
		push!(yout.dlkappa,     yin.dlkappa[i]     - zin.dlkappa[i]) 
		push!(yout.dleta_coeff, yin.dleta_coeff[i] - zin.dleta_coeff[i]) 
	end
	for i = 1:nPhi
		push!(yout.phix,  zin.phix[i] - yin.phix[i])
		push!(yout.Dphix, zin.Dphix[i]- yin.Dphix[i]) 
		push!(yout.dlphix,  zin.dlphix[i] - yin.dlphix[i])
		push!(yout.dlDphix, zin.dlDphix[i]- yin.dlDphix[i]) 
	end
	yout
end 

# defines +, -, *, / between Flow and Float64
for opt = (:+, :-, :*, :/) 
	@eval begin
		function ($opt)(yin::Flow, a::Real)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = Flow(yin.dim, 0, 0)
			for i = 1:nKap
				push!(yout.kappa, ($opt)(yin.kappa[i],a)) 
				push!(yout.eta_coeff, ($opt)(yin.eta_coeff[i],a)) 
			end
			for i = 1:nPhi
				push!(yout.phix, ($opt)(yin.phix[i],a))
				push!(yout.Dphix, ($opt)(yin.Dphix[i],a)) 
			end
			yout
		end # end function definition
	end # end the begin block
end # end for loop over functions

# +, -, * are associative
for opt = (:+, :-, :*)
	@eval ($opt)(a::Real, yin::Flow) = ($opt)(yin::Flow, a::Float64)
end
# ytest1 = Flow(2, 3)
# ytest2 = ytest1 + 5.7
# ytest3 = ytest2 * 1.7 + 4


for opt = (:+, :-, :*, :/) # this defines +, -, *, / between TrFlow and Float64
	@eval begin
		function ($opt)(yin::TrFlow, a::Real)
			nKap = length(yin.kappa)
			nPhi = length(yin.phix)
			yout = TrFlow(yin.dim, 0, 0)
			for i = 1:nKap
				push!(yout.kappa, ($opt)(yin.kappa[i],a)) 
				push!(yout.eta_coeff, ($opt)(yin.eta_coeff[i],a)) 
				push!(yout.dlkappa, ($opt)(yin.dlkappa[i],a)) 
				push!(yout.dleta_coeff, ($opt)(yin.dleta_coeff[i],a)) 
			end
			for i = 1:nPhi
				push!(yout.phix, ($opt)(yin.phix[i],a))
				push!(yout.Dphix, ($opt)(yin.Dphix[i],a)) 
				push!(yout.dlphix, ($opt)(yin.dlphix[i],a))
				push!(yout.dlDphix, ($opt)(yin.dlDphix[i],a)) 
			end
			yout
		end # end function definition
	end # end the begin block
end # end for loop over functions

# +, -, * are associative
for opt = (:+, :-, :*)
	@eval ($opt)(a::Real, yin::TrFlow) = ($opt)(yin::TrFlow, a::Float64)
end


# these are used in ode code
function norm_inf_any{D}(y::Array{Float64,D})
    norm(y,Inf)
end

function norm_inf_any(y::Flow)
    nKap = length(y.kappa)
    nPhi = length(y.phix)
    m=-Inf
    for i=1:nKap    
        m = max(norm_inf_any(y.kappa[i]), m)
        m = max(norm_inf_any(y.eta_coeff[i]), m)
    end
    for i=1:nPhi
        m = max(norm_inf_any(y.phix[i]), m)
        m = max(norm_inf_any(y.Dphix[i]), m)
    end
    m
end

function norm_inf_any(y::TrFlow)
    nKap = length(y.kappa)
    nPhi = length(y.phix)
    m=-Inf
    for i=1:nKap    
        m = max(norm_inf_any(y.kappa[i]), m)
        m = max(norm_inf_any(y.eta_coeff[i]), m)
        m = max(norm_inf_any(y.dlkappa[i]), m)
        m = max(norm_inf_any(y.dleta_coeff[i]), m)
    end
    for i=1:nPhi
        m = max(norm_inf_any(y.phix[i]), m)
        m = max(norm_inf_any(y.Dphix[i]), m)
        m = max(norm_inf_any(y.dlphix[i]), m)
        m = max(norm_inf_any(y.dlDphix[i]), m)
    end
    m
end


