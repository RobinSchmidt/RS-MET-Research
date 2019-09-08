clear all;

% assign the scale-factors for the parameters - these factors will be
% multiplied with the original parameters to achieve the desired
% formant-transformation:
formantGainScale = [1.00; 1.00; 1.00; 1.00; 1.00; 1.00];
formantFreqScale = [1.00; 1.50; 1.00; 1.00; 1.00; 1.00];
formantBwScale   = [1.00; 1.00; 1.00; 1.00; 1.00; 1.00];

% assign some more parameters:
preEmph             = true;  % switches pre-emphasis on and off
discardAntiformants = false; % if true, the antiformants will be ignored in
                             % the resythesis

% assign the visualization-parameters:
blockSize           = 8192;
fftSize             = blockSize;
plotRange           = 1:fftSize/8;

% read the input signal:
[inputSignal, sampleRate] = wavread('signals\Ahh.wav');

% take only a chunk (from somwhere in the middle):
signalBlock = inputSignal(10000:10000+blockSize-1);

% apply a hanning window:
windowedBlock = signalBlock.*hann(blockSize);

% load the parameters of the EQ-chain which are assumed to be optimized,
% such that the equalizers frequency-response approximates the signals
% spectral envelope:
preEmphCoeff = load('data\preEmphasisCoeffAhh.mat',    '-ASCII');
parameters   = load('data\eqParametersAhh5.1.mat',     '-ASCII'); 
 % 5.1 means 5 formants + 1 antiformant

% apply pre-emphasis filter:
if(preEmph)
 coeffs        = [1; preEmphCoeff];
 inputSignal   = filter(coeffs, 1, inputSignal);
 windowedBlock = filter(coeffs, 1, windowedBlock); 
end

%--------------------------------------------------------------------------
% the actual formant-transformation:

% decompose the parameter-vetor of the eq-chain:
numStages = round((length(parameters)-1)/3);
G         = parameters(1);
g         = parameters(2:(numStages+1));
OmegaC    = parameters((numStages+2):(2*numStages+1));
gamma     = parameters((2*numStages+2):(3*numStages+1)); 
bwOct     = gammaToBwOct(gamma, OmegaC);

% apply the desired transformation to the parameters:
g_new      = g.*formantGainScale;
if(discardAntiformants)
 for s=1:numStages
  if (g(s) < 1)
   g_new(s) = 1;  
  end 
 end
end
OmegaC_new = OmegaC.*formantFreqScale;
bwOct_new  = bwOct.*formantBwScale;
gamma_new  = bwOctToGamma(bwOct_new, OmegaC_new);

% compose the new parameter-vector:
parameters_new = [G; g_new; OmegaC_new; gamma_new];

% apply the transformation to the signal:
transformedSignal = transformFormantsViaEqualizer(inputSignal, ...
                     parameters, parameters_new);
transformedBlock  = transformFormantsViaEqualizer(windowedBlock, ...
                     parameters, parameters_new);

%--------------------------------------------------------------------------
% visualization:

% calculate the bin-frequencies
binFreqs = 0:(fftSize)-1;
binFreqs = binFreqs * sampleRate/fftSize;

% calculate the magnitude-spectrum of the input- and output-block:
X = abs(fft(windowedBlock));
Y = abs(fft(transformedBlock));

% normalize:
normalizer = 1/max(X);
X          = X.*normalizer;
Y          = Y.*normalizer;

% estimate the fundamental frequency:
fundamental = estimateFundamental(windowedBlock, sampleRate, 50, 300);

% get the spectral envelope of the input and output signal(block):
[X_env, C, P_c] = spectralEnvelopeViaTE(X, sampleRate, fundamental);
X_dB     = 20 * log10(X);
X_env_dB = 20 * log10(X_env);

[Y_env, C, P_c] = spectralEnvelopeViaTE(Y, sampleRate, fundamental);
Y_dB     = 20 * log10(Y);
Y_env_dB = 20 * log10(Y_env);

% get the original and transformed models frequency-response:
H        = generateEqualizerCurve(fftSize/2, parameters);
H_dB     = 20 * log10(H);
H_new    = generateEqualizerCurve(fftSize/2, parameters_new);
H_dB_new = 20 * log10(H_new);

% plot the output-spectrum and the original and transformed model-spectrum:
figure;
plot(binFreqs(plotRange), Y_dB(plotRange), 'k', ... 
     binFreqs(plotRange), H_dB_new(plotRange), 'k', ...
     binFreqs(plotRange), H_dB(plotRange), 'k--');
 axis([0, max(binFreqs(plotRange)), -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');
 legend('Ausgangsspektrum', 'modifizierter EQ-Frequenzgang', ...
        'originaler EQ-Frequenzgang' );
 
%--------------------------------------------------------------------------
% undo pre-emphasis and play:

if(preEmph)
 coeffs            = [1; preEmphCoeff];
 inputSignal       = filter(1, coeffs, inputSignal);
 transformedSignal = filter(1, coeffs, transformedSignal); 
 windowedBlock     = filter(1, coeffs, windowedBlock); 
end

soundsc(transformedSignal, sampleRate);



