function error = calculateErrorFunction(parameters, X, w)

% This function calculates the value of the error-function for some given
% set of equalizer-parameters, a given spectral envelope and a given
% error-weighting function.
%
% usage: 
% error = calculateErrorFunction(parameters, X, w)
%
% input-variables:
%  -parameters: the vector of equalizer-parameters [G; g; OmegaC; gamma]
%  -X: the spectral envelope of the input signal (from 0-pi - that is:
%      the redundant bins are already cut off)
%  -w: the error-weighting function (again, from 0-pi)
%
% output-variables:
%  -error: the value of the error-function

%--------------------------------------------------------------------------

% determine the number of bins:
numBins = length(X);

% calculate the magnitude response of the equalizer for the given set of
% parameters:
H = generateEqualizerCurve(numBins, parameters);

% calculate the error-function:
error = 2*mean( w.*( ( log(X.^2) - log(H.^2) ).^2) );