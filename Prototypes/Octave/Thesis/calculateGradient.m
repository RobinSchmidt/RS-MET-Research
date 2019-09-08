function gradient = calculateGradient(equalizerParameters, X, w, ... 
                                      minBin, maxBin)

% This function calculates the gradient of the error-function for some 
% given set of equalizer-parameters, a given spectral envelope and a 
% given error-weighting function.
%
% usage: 
%  gradient = calculateGradient(equalizerParameters, X, w, ...
%                               minBin, maxBin)
%
% input-variables:
%  -equalizerParameters: the vector of equalizer-parameters 
%                        [G; g; OmegaC; gamma]
%  -X: the spectral envelope of the input signal (from 0-pi - that is:
%      the redundant bins are already cut off)
%  -w: the error-weighting function (again, from 0-pi)
%  -minBin: lowest bin for which the error-weighting is nonzero
%  -maxBin: highest bin for which the error-weighting is nonzero
%
% output-variables:
%  -gradient: the gradient of the error-function
%             [dE_dG; dE_dg; dE_dOmegaC; dE_dgamma]

%--------------------------------------------------------------------------

% extract the number of bins:
numBins = length(X);

if(nargin<5)
 maxBin = numBins;
end
if(nargin<4)
 minBin = 1; 
end
% use a unity weighting function, if no weighting function is passed:
if(nargin<3)
 w = ones(numBins, 1); 
end

% extract the number of filter-stages:
numStages = round((length(equalizerParameters)-1)/3);

% decompose the parameter-vector:
G         = equalizerParameters(1);
g         = equalizerParameters(2:(numStages+1));
OmegaC    = equalizerParameters((numStages+2):(2*numStages+1));
gamma     = equalizerParameters((2*numStages+2):(3*numStages+1));

%--------------------------------------------------------------------------
% allocate and calculate the a- and b-coefficients for all the filter
% stages:
 b0 = (1+gamma.*sqrt(g))./(1+gamma./sqrt(g));
 b1 = -2*cos(OmegaC)./(1+gamma./sqrt(g)); 
 b2 = (1-gamma.*sqrt(g))./(1+gamma./sqrt(g));
 a0 = ones(numStages,1);
 a1 = b1;
 a2 = (1-gamma./sqrt(g))./(1+gamma./sqrt(g));
 
 
%--------------------------------------------------------------------------
% initialize the derivatives of the error function (with respect to the
% parameters of the eq-stages) with zeros, then accumulate them via a loop
% through the frequency-bins:
Num    = ones(numBins, numStages);  % numerator of the ma-squred spectrum
                                    % of a single eq-stage
Den    = ones(numBins, numStages);  % denominator 
H_sq_s = ones(numBins, numStages);  % mag-squared spectrum of a single eq-stage (index p)
H_sq   = ones(numBins, 1);          % mag-squared spectrum of the whole eq-chain

X_sq   = X.^2;                      % we want work with squared input 
                                    % magnitude-spectrum

% variables for the accumulation of the contributions of the individual
% bins to the derivative of E:
dE_dG  = 0;
dE_db0 = zeros(numStages, 1);
dE_db1 = zeros(numStages, 1);
dE_db2 = zeros(numStages, 1);
dE_da1 = zeros(numStages, 1);
dE_da2 = zeros(numStages, 1);
                                    
for k=minBin:maxBin
 
 % calculate the normalized radian frequency of this bin:
 OmegaK = pi*(k-1)/numBins;
 
 % these two cosines are needed at several places:
 cos1      =  2*cos(OmegaK);
 cos2      =  2*cos(2*OmegaK); 
 
 % calculate the frequency response of the equalizer-chain at this bin (as
 % the product of the individual stages) - thereby also store the respective
 % numerators and denominators (we need them later again): 
 H_sq(k) = 1.0; % multiplicatively accumulates the mag-squared spectra of the eq-stages
 for i=1:numStages
  Num(k,i) =   b0(i)^2 + b1(i)^2 + b2(i)^2 ...
             + cos1*( b0(i)*b1(i) + b1(i)*b2(i) ) ...
             + cos2*b0(i)*b2(i);
             
  Den(k,i) =   a0(i)^2 + a1(i)^2 + a2(i)^2 ...
             + cos1*( a0(i)*a1(i) + a1(i)*a2(i) ) ...
             + cos2*a0(i)*a2(i); 
   
  H_sq_s(k,i) =  Num(k,i)/Den(k,i);
  
  H_sq(k)     = H_sq(k) * H_sq_s(k,i);  
 end
 % apply the overall gain to the mag-squared spectrum:
 H_sq(k) = G^2 * H_sq(k);
 
 % calculate the contribution at this bin to the partial derivative of the
 % global gain factor G (and add it to the accumulating sum):
 dlnH_sq_dG = 2/G;
 dE_dG      = dE_dG + w(k) * ( log(H_sq(k)) - log(X_sq(k)) ) * dlnH_sq_dG;
 
 % calculate the contribution at this bin to the partial derivative of E
 % with repsect to b0(i):
 for i=1:numStages
  
  % calculate derivtive of H^2 with respect to the biquad-coefficients:
  dHsq_db0i =  (2*b0(i) + cos1*b1(i) + cos2*b2(i)) / Den(k,i);
  dHsq_db1i =  (2*b1(i) + cos1*(b0(i)+b2(i))     ) / Den(k,i);
  dHsq_db2i =  (2*b2(i) + cos1*b1(i) + cos2*b0(i)) / Den(k,i);  
  dHsq_da1i = -(2*a1(i) + cos1*(a0(i)+a2(i))     ) * Num(k,i)/(Den(k,i))^2;
  dHsq_da2i = -(2*a2(i) + cos1*a1(i) + cos2*a0(i)) * Num(k,i)/(Den(k,i))^2;  
 
  % accumulate:  
  factor    = w(k)*(log(H_sq(k))-log(X_sq(k))) * (1/H_sq_s(k,i));
  dE_db0(i) = dE_db0(i) + factor*dHsq_db0i;
  dE_db1(i) = dE_db1(i) + factor*dHsq_db1i;  
  dE_db2(i) = dE_db2(i) + factor*dHsq_db2i;    
  dE_da1(i) = dE_da1(i) + factor*dHsq_da1i;  
  dE_da2(i) = dE_da2(i) + factor*dHsq_da2i;    
  
 end % end of "for i=1:numStages"
 
end % end of "for k=minBin:maxBin"

% apply the constant factor (4/numBins):
dE_dG  = 4*dE_dG/numBins;
dE_db0 = 4*dE_db0./numBins;
dE_db1 = 4*dE_db1./numBins;
dE_db2 = 4*dE_db2./numBins;
dE_da1 = 4*dE_da1./numBins;
dE_da2 = 4*dE_da2./numBins;


%--------------------------------------------------------------------------
% calculate the derivatives of the a- and b-coefficients with respect to the 
% equalizer parameters for all the filter stages:

 % calculate the partial derivatives of the coefficients for stage i: 
 factor  = 1./(1 + gamma./sqrt(g)).^2; % this is a common factor in many  
                                       % inner partial derivatives
 
 db0_dg = factor.* ( 0.5*gamma./sqrt(g) + gamma.^2./g + 0.5*gamma./sqrt(g.^3) );
 db1_dg = factor.* ( -gamma./sqrt(g.^3).* cos(OmegaC) );
 db2_dg = factor.* ( -0.5*gamma./sqrt(g) - gamma.^2./g + 0.5*gamma./sqrt(g.^3));
 da0_dg = zeros(numStages,1);
 da1_dg = db1_dg;
 da2_dg = factor.* ( gamma./sqrt(g.^3) );
 
 db0_dOmegaC = zeros(numStages,1);
 db1_dOmegaC = (2.*sin(OmegaC))./ (1+gamma./sqrt(g));
 db2_dOmegaC = zeros(numStages,1);
 da0_dOmegaC = zeros(numStages,1);
 da1_dOmegaC = db1_dOmegaC;
 da2_dOmegaC = zeros(numStages,1);   
 
 db0_dgamma = factor.* ( sqrt(g) - (1./sqrt(g)) );
 db1_dgamma = factor.* ( 2*cos(OmegaC)./sqrt(g) );
 db2_dgamma = factor.* ( -sqrt(g) - (1./sqrt(g)) );
 da0_dgamma = zeros(numStages,1);
 da1_dgamma = db1_dgamma;
 da2_dgamma = factor.* ( -2./sqrt(g) );

% now that we know the derivatives of the error-function with respect to
% the biquad-coefficients as well as the derivatives of the
% biquad-coefficients with respect to the equalizer parameters, we can
% calculate the derivates of the error-function with respect to the
% equalizer-parameters by means of the generalized chain-rule:
dE_dg = dE_db0.*db0_dg + dE_db1.*db1_dg + dE_db2.*db2_dg + ...
        dE_da1.*da1_dg + dE_da2.*da2_dg;

dE_dOmegaC = dE_db0.*db0_dOmegaC + dE_db1.*db1_dOmegaC + ...
             dE_db2.*db2_dOmegaC + dE_da1.*da1_dOmegaC + dE_da2.*da2_dOmegaC;

dE_dgamma = dE_db0.*db0_dgamma + dE_db1.*db1_dgamma + dE_db2.*db2_dgamma + ...
            dE_da1.*da1_dgamma + dE_da2.*da2_dgamma;

% combine the partial derivatives to the gradient vector:
gradient = [dE_dG; dE_dg; dE_dOmegaC; dE_dgamma];
