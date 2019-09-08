function [w, minBin, maxBin] = calculateErrorWeighting(X, sampleRate, ...
                                f_min, f_max, exponent)

% This function calculates an appropriate error-weighting function for the
% cost-function. The function will be bandpass-like (nonzero only in
% between f_min, f_max) and inside this band it will depend on the spectral
% envelope like w = X.^exponent which gives a constant for exponent == 0.
% Higher exponents will give rise to a higher error-weighting at spectral
% peaks, which in turn forces the fit of the eq-curve to the spectral
% envelope to be better at those spectral peaks. It seems resonable to
% choose the exponent somewhere between 0...2.
%
% usage: 
%  w = calculateErrorWeighting(X, sampleRate, f_min, f_max, exponent)
%
% input-variables:
%  -X: the spectral envelope of the input signal (from 0-pi - that is:
%      the redundant bins are already cut off)
%  -sampleRate: the input signals sample-rate
%  -f_min: minimum frequency (in Hz) at which we need a good fit
%  -f_max: maximum frequency (in Hz) at which we need a good fit
%  -exponent: determines, in which way the weighting should depend on the
%             spectral envelope itself (see above)
%
% output-variables:
%  -w: the error-weighting function (again, from 0-pi)
%  -minBin: minimum bin with nonzero weight
%  -maxBin: maximum bin with nonzero weight

%--------------------------------------------------------------------------

% extract the number of bins:
numBins = length(X);

% calculate the normalized radian bin-frequencies:
Omegas = (0:1:(numBins-1))';
Omegas = pi*Omegas/numBins;

% use X.^exponent as the weighting-function:
w = X.^exponent;

% calculate the minimum and maximum bin where the weighting should be
% nonzero and set the weighting function to zero outside this interval:
minBin = floor(2*numBins*f_min/sampleRate);
maxBin = ceil(2*numBins*f_max/sampleRate);
w(1:minBin-1)          = 0;
w(maxBin+1:numBins)    = 0;

% normalize the weighting, such that the integral (sum) becomes unity times
% the number of nonzero bins (i.e. the sum is always the same as for a 
% unity weighting function in the frequency-range of interest):
w = w.*(maxBin-minBin+1)/sum(w);