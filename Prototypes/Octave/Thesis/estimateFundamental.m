function fundamental = estimateFundamental(inputSignal, sampleRate, ...
                                           minFundamental, maxFundamental)

% This function estimates the fundamental frequency of the input signal via
% a simple autocorrelation approach.
%
% usage: 
%  fundamental = estimateFundamental(inputSignal, sampleRate, ...
%                                    minFundamental, maxFundamental)
%
% input-variables:
%  -inputSignal: the input signal
%  -sampleRate: the input signals sample-rate
%  -minFundamental: minimal expected fundamental frequency (in Hz)
%  -maxFundamental: maximal expected fundamental frequency (in Hz)
%
% output-variables:
%  -fundamental: estimated fundamental frequency (in Hz)

%--------------------------------------------------------------------------

% calculate the minimum and maximum lag at which the maximum
% (corresponding to the period) is expected:
minLag   = floor(sampleRate / maxFundamental);
maxLag   = ceil(sampleRate  / minFundamental);

% calculate the autocorrelation function of the signal:
inputAcf = xcorr(inputSignal, maxLag, 'coeff'); % 'coeff' normlizes the acf

% keep only the right half (the acf is symmetric):
inputAcf = inputAcf(maxLag+1:length(inputAcf));

% extract the peaks (other than the one at lag=0) of the acf:
inputAcfPeaks = zeros(length(inputAcf),1);
for k=minLag:length(inputAcfPeaks)-1
 if( (inputAcf(k) > inputAcf(k-1)) & (inputAcf(k) > inputAcf(k+1)) )
  % copy the value from the acf:
  inputAcfPeaks(k) = inputAcf(k);
 end 
end

% select the largest of the autocorrelation-peaks and use it as estimate
% for the pitch period:
[dummy, periodInSamples] = max(inputAcfPeaks);

% index=1 corresponds to lag=0, so subtract 1:
periodInSamples = periodInSamples-1;

% -------------------------------------------------------------------------
% refine the estimated period in samples by fitting a quadratic parabola
% through the points periodInSamples-1, periodInSamples, periodInSamples+1
% and use the location of the maximum of this parabola as new
% periodInSamples:

% calculate polynomial coefficients for the parabola:
t_0 = periodInSamples+1; % the reference time-index
a_0 = inputAcf(t_0);
a_1 = 0.5 * ( inputAcf(t_0+1) -  inputAcf(t_0-1) );
a_2 = inputAcf(t_0+1) - a_1 - a_0;

% calculate the location of the maximum of this parabola:
refinedPeriodInSamples = (2*a_2*t_0-a_1)/(2*a_2);

% again, compensate for the fact that MatLab indices start at 1:
refinedPeriodInSamples = refinedPeriodInSamples-1;

% multiply by the sample-period to get the period in seconds:
periodInSeconds    = refinedPeriodInSamples/sampleRate;
fundamental        = 1/periodInSeconds;