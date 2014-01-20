#------------------------------------
#  1-d targets
#--------------------------------------------
function targetUnif1d(kappa::Array{Array{Float64,1}})
	TargSig = 0.05
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k =1:length(kappa)
		kap = kappa[k][1]
		if kap <= 0
			dentmp = exp(-((kap).^2.0)/(2.0*TargSig.^2.0))
			gradHtmp = (-((kap))/(TargSig.^2))  
		elseif 0 < kap < 1
			dentmp = 1.0
			gradHtmp = 0.0
		else
			dentmp = exp(-((kap - 1.0).^2.0)/(2.0*TargSig.^2.0))
			gradHtmp = (-((kap - 1.0))/(TargSig.^2.0))
		end
		push!(density, dentmp/(1.0+ TargSig*sqrt(2.0*pi)))
		push!(gradH,[gradHtmp])
	end
	density, gradH
end
function targetUnif2d(kappa::Array{Array{Float64,1}})
	TargSig = 0.05
	density = Float64[]
	gradH   = Array{Float64,1}[]
	for k =1:length(kappa)
		densityx = kappa[k][1] < 0.0 ? exp(-1.0 * kappa[k][1].^2.0 / (2.0*TargSig.^2)) :  kappa[k][1] < 1.0 ? 1.0 : exp(-1.0 * (kappa[k][1] - 1).^2.0 / (2.0*TargSig.^2) )
		densityy = kappa[k][2] < 0.0 ? exp(-1.0 * kappa[k][2].^2.0 / (2.0*TargSig.^2)) :  kappa[k][2] < 1.0 ? 1.0 : exp(-1.0 * (kappa[k][2] - 1).^2.0 / (2.0*TargSig.^2) )
		densityx /= 1.0 + TargSig * sqrt(2.0 * pi)
		densityy /= 1.0 + TargSig * sqrt(2.0 * pi)
		push!(density, densityx * densityy)
		gradHx =  kappa[k][1] < 0.0 ?  -1.0*kappa[k][1] / (TargSig.^2) : kappa[k][1] > 1.0  ?  -1.0*(kappa[k][1] - 1)/(TargSig.^2) : 0.0 
		gradHy =  kappa[k][2] < 0.0 ?  -1.0*kappa[k][2] / (TargSig.^2) : kappa[k][2] > 1.0  ?  -1.0*(kappa[k][2] - 1)/(TargSig.^2) : 0.0 
		push!(gradH,[gradHx,gradHy])
	end
	density, gradH
end


#------------------------------------
#  2-d targets
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
function hfun32d(DefXnew,DefYnew)
	sigma=.5
	gradx= -(DefXnew)/(sigma^2)   
	grady= -(DefYnew)/(sigma^2) 
	expH= exp(-(DefXnew.^2+DefYnew.^2)/(2*sigma^2)  )/(2*pi*sigma.^2)
	expH, gradx, grady
end

