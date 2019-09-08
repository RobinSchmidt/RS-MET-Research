function arCoeffs = telpc(signal, order, minFundamental, ...
                          maxFundamental, sampleRate);

% This function estimates the coefficicients of an AR-model by means of the
% True-Envelope Linear-Prediction (TE-LPC) method. Its usage is similar to
% the conventional lpc-function.
%
% usage: 
%  arCoeffs = telpc(signal, order, minFundamental, ...
%                   maxFundamental, sampleRate);
%
% input-variables:
%  -signal: the input-signal (usually a windowed block thereof)
%  -order: order of the AR-model to be estimated
%  -minFundamental (optional): minimum expected fundamental frequency of
%    the signal in Hz (needed for fundamental frequency estimation)
%  -maxFundamental (optional): maximum expected fundamental frequency of
%    the signal in Hz (needed for fundamental frequency estimation)
%  -sampleRate (optional): the sample-rate of the signal (needed to make
%    sense of the minFundamental and maxFundamental values in terms of
%    normalized frequencies)
%
% output-variables:
%  -arCoeffs: vector of the AR-coefficients in the same format as the
%    conventional lpc-function returns them

%--------------------------------------------------------------------------

% assign default values for the optional input-arguments:
if( nargin<5 )
 sampleRate = 44100; 
end
if( nargin<4 )
 maxFundamental = 2000;
end
if( nargin<3 )
 minFundamental = 50; 
end

% estimate the fundamental frequency of the input signal (in Hz):
fundamental = estimateFundamental(signal, sampleRate, ... 
                                  minFundamental , maxFundamental )
                                 
% calculate the fft-magnitude-spectrum of the input signal:
X = abs(fft(signal));                                 

% get the spectral envelope via the true-envelope algorithm:
X_env = spectralEnvelopeViaTE(X, sampleRate, fundamental);

% calculate the autocorrelation-function as the inverse fourier-transform
% of the squared input-envelope:
acf = real(ifft(X_env.^2));

% calculate the AR-coefficients from this autocorrelation-function:
arCoeffs = levinson(acf, order);