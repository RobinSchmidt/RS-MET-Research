clear all;

% parameter settings:
blockSize           = 2048;
fftSize             = blockSize;
preEmph             = true;
minFundamental      = 50;
maxFundamental      = 300;
envSmoothing        = 1.0;     % should be <=1 - the smaller, the smoother
minFormantCenter    = 600;
maxFormantCenter    = 4000;
numFormants         = 5;
numAntiformants     = 1;
numStages           = numFormants + numAntiformants;
errorWeightExponent = 1.0;

filenamePreEmphCoeff = 'data\preEmphasisCoeffAhh.mat'; 
filenameEqParams     = 'data\eqParametersAhh5.1.mat';  % 5.1 means 5 formants + 1 antiformant
filenameEqInitParams = 'data\eqInitParametersAhh5.1.mat'; 

% read the input signal:
[inputSignal, sampleRate] = wavread('signals\Ahh.wav');

% take only a chunk (from somwhere in the middle):
signalBlock = inputSignal(10000:10000+blockSize-1);

% apply a hanning window:
windowedBlock = signalBlock.*hann(blockSize);

% apply pre-emphasis:
preEmphCoeff = 0;
if(preEmph)
 % get the optimal pre-emphasis coefficient (the optimal one-step
 % predictor):
 coeffs       = real(lpc(windowedBlock, 1));
 preEmphCoeff = coeffs(2);
 % apply pre-emphasis to the signal:
 inputSignal   = filter(coeffs, 1, inputSignal);
 windowedBlock = filter(coeffs, 1, windowedBlock); 
end

%pause;

% estimate the fundamental frequency of the input signal (in Hz):
fundamental = estimateFundamental(windowedBlock, sampleRate, minFundamental, maxFundamental);

% calculate the FFT-magnitude-spectrum:
X = abs(fft(windowedBlock));

% normalize:
normalizer = 1/max(X);
X          = normalizer * X;

% estimate the spectral envelope via the true envelope algorithm:
X_env = spectralEnvelopeViaTE(X, sampleRate, fundamental, 1.0, 0.8, 1.0, envSmoothing);

% X = estimateTrueSpectralEnvelope(windowedBlock, sampleRate, fftSize, fundamental, envSmoothing);
% X = X(1:length(X)/2);
% X = X./max(X);

% for the following optimization-procedure, we use only the first half of
% the fft-spectrum, that is: the redundant bins are cut-off:
X     = X(1:length(X)/2);
X_env = X_env(1:length(X_env)/2);

%--------------------------------------------------------------------------

[w, minBin, maxBin] = calculateErrorWeighting(X_env, sampleRate, ... 
                       minFormantCenter, maxFormantCenter, ...
                       errorWeightExponent);
%plot(w);
%pause;                     
                      
parameters = findInitialEqualizerParameters(X_env, w, numFormants, numAntiformants);

% save the initial equalizer-parameters to a .mat file:
save(filenameEqInitParams, 'parameters', '-ASCII', '-DOUBLE');

% calculate the normalized radian bin-frequencies:
numBins = length(X_env);
Omegas  = (0:1:(numBins-1))';
Omegas  = pi*Omegas/numBins;

% refine the error weighting, such that the minimum bin with nozero weight
% corresponds to a frequency one octave below the lowest eq
% center-frequency and the maxnimum bin with nozero weight
% corresponds to a frequency one octave above the highes eq
% center-frequency:
OmegaC = parameters((numStages+2):(2*numStages+1));
OmegaC = OmegaC(1:numFormants); % truncate the Omegas of the antiformant-bands
f_min  = floor( (1/1.5) * OmegaC(1)*sampleRate/(2*pi) );
f_max  = ceil(   1.5    * OmegaC(numFormants)*sampleRate/(2*pi) );
[w, minBin, maxBin] = calculateErrorWeighting(X_env, sampleRate, f_min, f_max, errorWeightExponent);

%parameters          = findInitialEqualizerParameters(X_env, w, numFormants, numAntiformants);

% calculate the initial eq-curve (before optimization):
H_init = generateEqualizerCurve(numBins, parameters);

% optimize the eq-parameters:
%parameters = optimizeEQParametersViaSCG(parameters, X_env, w);
parameters = optimizeEQParametersViaSCG(parameters, X_env, w, 100, sampleRate, fftSize/4);

% save the optimal pre-emphasis coefficient estimated equalizer-parameters
% to a .mat file:
% save(filename, 'preEmphCoeff', 'parameters', '-ASCII', '-DOUBLE');
save(filenamePreEmphCoeff, 'preEmphCoeff', '-ASCII', '-DOUBLE');
save(filenameEqParams,     'parameters',   '-ASCII', '-DOUBLE');



% generate the equalizers frequency response curve:
H = generateEqualizerCurve(numBins, parameters);

% plot the test-curve, and the optimized eq-curve:
% calculate the bin-frequencies
binFreqs = 0:(fftSize/2)-1;
binFreqs = binFreqs * sampleRate/fftSize;

X_dB     = 20 * log10(X);
X_env_dB = 20 * log10(X_env);
H_dB     = 20 * log10(H);

figure;
 plot(binFreqs(1:fftSize/4), X_dB(1:fftSize/4), ... 
      binFreqs(1:fftSize/4), X_env_dB(1:fftSize/4), ... 
      binFreqs(1:fftSize/4), H_dB(1:fftSize/4)); 


