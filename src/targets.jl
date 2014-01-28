#------------------------------------
#  Uniform targets
#--------------------------------------------
function targetUnif1d(kappa::Array{Array{Float64,1},1})
	TargSig = 0.25 # this controls the decay of the smooth trucation
	Limit = 1.0
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k =1:length(kappa)
		kap = kappa[k][1]
		if kap <= -Limit
			dentmp = exp(-((kap + Limit).^2.0)/(2.0 * TargSig * TargSig))
			gradHtmp = (-((kap + Limit))/(TargSig.^2.0)) 
		elseif -Limit < kap < Limit
			dentmp = 1.0
			gradHtmp = 0.0
		else
			dentmp = exp(-((kap - Limit).^2.0)/(2.0*TargSig.^2.0))
			gradHtmp = (-((kap - Limit))/(TargSig.^2.0))
		end
		push!(density, dentmp/(Limit + Limit + TargSig*sqrt(2.0*pi)))
		push!(gradH,[gradHtmp])
	end
	density, gradH
end
function targetUnif2d(kappa::Array{Array{Float64,1},1})
	N = length(kappa)
	densitx, gradHx = targetUnif1d(Array{Float64,1}[[kappa[i][1]] for i=1:N] )
	density, gradHy = targetUnif1d(Array{Float64,1}[[kappa[i][2]] for i=1:N] )
	densitx.*density, Array{Float64,1}[[gradHx[i][1], gradHy[i][1]] for i=1:N] 
end
function targetUnif3d(kappa::Array{Array{Float64,1},1})
	N = length(kappa)
	densitx, gradHx = targetUnif1d(Array{Float64,1}[[kappa[i][1]] for i=1:N] )
	density, gradHy = targetUnif1d(Array{Float64,1}[[kappa[i][2]] for i=1:N] )
	densitz, gradHz = targetUnif1d(Array{Float64,1}[[kappa[i][2]] for i=1:N] )
	densitx .* density .* densitz, Array{Float64,1}[[gradHx[i][1], gradHy[i][1], gradHz[i][1]] for i=1:N]
end 


#------------------------------------
#  Normal Targets
#--------------------------------------------
function targetNormal1d(kappa::Array{Array{Float64,1},1})
	targsig = 1.0
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k = 1:length(kappa)
		push!(density, exp(-1.0 * kappa[k][1] .* kappa[k][1] / (2.0 * targsig * targsig)  ) / (sqrt(2.0 * pi) * targsig) )
		push!(gradH,[-kappa[k][1]/ (targsig * targsig)]) 
	end
	density, gradH
end
function targetNormal2d(kappa::Array{Array{Float64,1},1})
	N = length(kappa)
	densitx, gradHx = targetNormal1d(Array{Float64,1}[[kappa[i][1]] for i=1:N] )
	density, gradHy = targetNormal1d(Array{Float64,1}[[kappa[i][2]] for i=1:N] )
	densitx.*density, Array{Float64,1}[[gradHx[i][1], gradHy[i][1]] for i=1:N]
end 
function targetNormal3d(kappa::Array{Array{Float64,1},1})
	N = length(kappa)
	densitx, gradHx = targetNormal1d(Array{Float64,1}[[kappa[i][1]] for i=1:N] )
	density, gradHy = targetNormal1d(Array{Float64,1}[[kappa[i][2]] for i=1:N] )
	densitz, gradHz = targetNormal1d(Array{Float64,1}[[kappa[i][2]] for i=1:N] )
	densitx .* density .* densitz, Array{Float64,1}[[gradHx[i][1], gradHy[i][1], gradHz[i][1]] for i=1:N]
end 



# function targetNormal2d(kappa::Array{Array{Float64,1},1})
# 	targsig = 1.0
# 	density = Float64[]
# 	gradH   = Array{Float64,1}[]
# 	for k = 1:length(kappa)
# 		push!(density, exp(-(kappa[k][1].^2 + kappa[k][2].^2) / (2 * targsig^2)  ) / (2 * pi * targsig^2) )
# 		push!(gradH,[-kappa[k][1]/(targsig^2) , -kappa[k][2] / (targsig^2)]) 
# 	end
# 	density, gradH
# end



#------------------------------------
# The following is old code and needs to be ported to Julia 
#--------------------------------------------
function targetGaussianMixture2d(x, p,mu1,mu2,sigma1,sigma2)
	den=  p*normpdf(x,mu1,sigma1) + (1-p)*normpdf(x,mu2,sigma2)
	dden1=(  p)*normpdf(x,mu1,sigma1).*(-(x-mu1)/sigma1^2)
	dden2=(1-p)*normpdf(x,mu2,sigma2).*(-(x-mu2)/sigma2^2)
	# dden=den.*(-(kap-mu)/TargSig^2)
	gradH=(dden1+dden2)./den
	den, gradH
end

function hfun12d(DefXnew,DefYnew)
	n=2
	sigma=.5
	gradx=(-n/2)*(1/sigma).*(2*DefXnew/sigma)./(1+(DefXnew/sigma).^2)
	grady=(-n/2)*(1/sigma).*(2*DefYnew/sigma)./(1+(DefYnew/sigma).^2)
	
	expH=(1/sigma^2)*(gamma(n/2)/(sqrt(pi)*gamma((n-1)/2)))^2 * (1+(DefXnew/sigma).^2).^(-n/2).*(1+(DefYnew/sigma).^2).^(-n/2)
	expH, gradx, grady
end
function hfun22d(DefXnew,DefYnew)
	n=2
	sigma=3
	sigma2=1
	gradx=(-n/2)*(1/sigma).*(2*DefXnew/sigma)./(1+(DefXnew/sigma).^2)
	grady=(-1/sigma2^2).*(DefYnew)
	
	expH=(1/sigma)*(gamma(n/2)/(sqrt(pi)*gamma((n-1)/2))) * (1+(DefXnew/sigma).^2).^(-n/2).*(1/(sqrt(2*pi)*sigma2)).*(exp(-DefYnew.^2/(2*sigma^2)))
	expH, gradx, grady
end

