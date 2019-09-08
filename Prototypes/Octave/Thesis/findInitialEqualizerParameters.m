function parameters = findInitialEqualizerParameters(X, w, ...
                                                     numPeakStages, ... 
                                                     numValleyStages)

% This function calculates an initial guess for the equalizer-parameters to
% be used as initialization for the optimization procedure.
%
% usage: 
%  parameters = findInitialEqualizerParameters(X, numStages, sampleRate, w)
%
% input-variables:
%  -X: the spectral envelope of the input signal (from 0-pi - that is:
%      the redundant bins are already cut off)
%  -w: the error-weighting function (again, from 0-pi) - eq-bands will be
%      put only inside the nonzero portion of this function
%  -numPeakStages: number of the equalizer stages with g > 1 (for
%    formants)
%  -numValleyStages (optional, default=0): number of the equalizer stages
%    with g < 1 (for antiformants)
%
% output-variables:
%  -parameters: the vector of the initial equalizer-parameters 
%   [G; g; OmegaC; gamma]

%--------------------------------------------------------------------------

% assign default values for optional parameters:
if( nargin<4 )
 numValleyStages = 0; 
end

% extract the number of bins:
numBins = length(X);

% find the minimum and maximum bin for which the weighting-function is
% nonzero:
minBin = 1;
k      = 1;
while( w(k) <= 0 )
 k = k+1; 
end
minBin = k

maxBin = numBins;
k      = numBins;
while( w(k) <= 0 )
 k = k-1; 
end
maxBin = k

% calculate the normalized radian bin-frequencies:
Omegas = (0:1:(numBins-1))';
Omegas = pi*Omegas/numBins;

%--------------------------------------------------------------------------
% find the initial parameters for the formant-eq-bands (those with g > 1):

% we only look at the maxima in the spectral envelope as candidates for the
% peak-eq center-frequencies:
maxima   = zeros(length(X),1);
for k=minBin+1:maxBin-1
 if( X(k)>=X(k-1) && X(k)>=X(k+1) )
  maxima(k) = X(k);  
 else
  maxima(k) = 0;
 end 
end

% keep only the "numPeakStages" highest peaks:
numPeaks = 0;     % counts, how many peaks already have been found
peaks    = zeros(length(maxima), 1);
while( numPeaks < numPeakStages )
 [maxValue, maxIndex] = max(maxima);
 if( maxValue == 0 )
  disp('warning: less peaks that peak-eq-bands');
  break;  
 end
 peaks(maxIndex,1)  = maxValue;
 maxima(maxIndex,1) = 0;
 numPeaks           = numPeaks + 1;
end

% at these "numPeakStages" largest peaks we put the center-frequencies of our
% peak-eq-stages
tmp = X(minBin:maxBin);
%G   = mean(tmp);               % arithmetic mean
G   = exp(mean(log(tmp)))      % gemoetric mean
for p=1:numPeakStages
 for k=minBin:maxBin
  if( peaks(k) ~= 0 )
   OmegaC(p)  = Omegas(k);
   g(p)       = 1.0*peaks(k)/G;
   %bw         = 1/8;
   %gamma(p) = sinh(0.5*log(2)*bw*OmegaC(p)/sin(OmegaC(p)))*sin(OmegaC(p));
   gamma(p)   = 0.005; 
   peaks(k,1) = 0;
   break;
  end  
 end
end

%--------------------------------------------------------------------------
% find the initial parameters for the antiformant-eq-bands (those with 
% g < 1):

% we only look at the minima in the spectral envelope as candidates for the
% valley-eq center-frequencies:
minima   = inf*ones(length(X),1);
for k=minBin+1:maxBin-1
 if( X(k)<=X(k-1) && X(k)<=X(k+1) )
  minima(k) = X(k);  
 else
  minima(k) = inf;
 end 
end

% keep only the "numValleyStages" lowest valleys:
numValleys = 0;     % counts, how many valleys already have been found
valleys    = inf*ones(length(minima), 1);
while( numValleys < numValleyStages )
 [minValue, minIndex] = min(minima);
 if( minValue == inf )
  disp('warning: less valleys that valley-eq-bands');
  break;  
 end
 valleys(minIndex,1) = minValue;
 minima(minIndex,1)  = inf;
 numValleys          = numValleys + 1;
end

% at these "numValleyStages" largest peaks we put the center-frequencies of
% our valley-eq-stages
for p=(numPeakStages+1):(numPeakStages+numValleyStages)
 for k=minBin:maxBin
  if( valleys(k) ~= inf )
   OmegaC(p)    = Omegas(k);
   g(p)         = 1.0*valleys(k)/G;
   gamma(p)     = 0.005; 
   valleys(k,1) = inf;
   break;
  end  
 end
end

%--------------------------------------------------------------------------

% transpose the results to give row-vectors:
g      = g';
OmegaC = OmegaC';
gamma  = gamma';

% combine the vectors of grouped parameters into one single vector:
parameters = [G; g; OmegaC; gamma];