function freqResponse = generateEqualizerCurve(numBins, parameters);

% This function generates a vector which is the magnitude response of a
% chain of parametric equalizers. The vector of the equalizer-parameters is
% assumed to be in the form: [G; g; OmegaC; bw], where G is an overall 
% gain-factor, g is the vector of the N gain-factors at the eq-center
% frequencies, OmegaC is the vector of the N eq-center frequencies 
% (expressed as normalized radian frequency) and gamma is the vecctor of
% the N eq-bandwidth-parameters.
%
% usage: 
%  freqResponse = generateEqualizerCurve(numBins, parameters);
%
% input-variables:
%  -numBins: the number of frequency-bins for which the freq-response is 
%            to be calculated (the frequency-range is always 0-pi, so this
%            value should be half of the fft-size of the spectral envelope)
%  -parameters: the vector of equalizer-parameters [G; g; OmegaC; gamma]
%
% output-variables:
%  -freqResponse: the frequency-response of the eq-chain (from 0-pi, that
%                 is: without any redundant bins)

%--------------------------------------------------------------------------

% decompose the parameter-vector:
numStages = round((length(parameters)-1)/3);
G         = parameters(1);
g         = parameters(2:(numStages+1));
OmegaC    = parameters((numStages+2):(2*numStages+1));
gamma     = parameters((2*numStages+2):(3*numStages+1)); 

freqResponse = ones(numBins,1);

% calculate the normalized radian bin-frequencies:
Omegas = (0:1:(numBins-1))';
Omegas = pi*Omegas/numBins;

for i=1:numStages
 
 % calculate the biquad-filter coefficients (eq. 16):
 b0 = (1+gamma(i)*sqrt(g(i))) / (1+gamma(i)/sqrt(g(i)));
 a0 = 1;
 b1 = -2*cos(OmegaC(i))    / (1+gamma(i)/sqrt(g(i)));
 a1 = -2*cos(OmegaC(i))    / (1+gamma(i)/sqrt(g(i)));
 b2 = (1-gamma(i)*sqrt(g(i))) / (1+gamma(i)/sqrt(g(i)));
 a2 = (1-gamma(i)/sqrt(g(i))) / (1+gamma(i)/sqrt(g(i)));
 
 % calculate the frequency-response of this stage (with index i) and
 % multiply the old frequency response with it: 
 for k=1:numBins
  Omega           = Omegas(k);
  Num             = b0^2 + b1^2 + b2^2 + 2*cos(Omega)*(b0*b1 + b1*b2) + 2*cos(2*Omega) * b0*b2;
  Den             = a0^2 + a1^2 + a2^2 + 2*cos(Omega)*(a0*a1 + a1*a2) + 2*cos(2*Omega) * a0*a2; 
  if( Den~=0 )
   freqResponse(k) = freqResponse(k) * sqrt(Num/Den);
  else
   freqResponse(k) = realmax;
   disp('warning: Den==0 in generateEqualizerCurve');   
  end
 end  
 
end

% scale the frequency-response by an overall gain factor:
freqResponse = G * freqResponse;