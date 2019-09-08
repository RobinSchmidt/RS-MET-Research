clear all;

% parameter settings:
blockSize      = 8192;
fftSize        = blockSize;
plotRange      = 1:fftSize/8;
preEmph        = true;
cepstralWindow = 1;   % 0 for rectangular, 1 for hamming window


% load the parameters for the EQ-chain, the frequency response of which will
% be plotted:
eqParamsOptimized = load('data\eqParametersAhh5.1.mat',     '-ASCII'); % 5.1 means 5 formants + 1 antiformant
eqParamsInitial   = load('data\eqInitParametersAhh5.1.mat', '-ASCII');

% read the input signal:
[inputSignal, sampleRate] = wavread('signals\Ahh.wav');

% calculate the bin-frequencies
binFreqs = 0:(fftSize)-1;
binFreqs = binFreqs * sampleRate/fftSize;

% apply pre-emphasis filter:
preEmphCoeff = 0;
if(preEmph)
 % get the optimal pre-emphasis coefficient (the optimal one-step
 % predictor):
 coeffs       = real(lpc(inputSignal, 1));
 preEmphCoeff = coeffs(2);
 % apply pre-emphasis to the signal:
 inputSignal = filter(coeffs, 1, inputSignal);
end

% take only a chunk (from somwhere in the middle):
signalBlock = inputSignal(10000:10000+blockSize-1);

% apply a hanning window:
windowedBlock = signalBlock.*hann(blockSize);

% calculate the magnitude-spectrum of the block:
X = abs(fft(windowedBlock));

% normalize:
normalizer = 1/max(X);
X          = X.*normalizer;

% estimate the fundamental frequency:
fundamental = estimateFundamental(windowedBlock, sampleRate, 50, 300);

% get the spectral envelope:
[X_env, C, P_c] = spectralEnvelopeViaTE(X, sampleRate, fundamental, ... 
                                        1.0, 0.8, cepstralWindow, 1.0);
X_dB     = 20 * log10(X);
X_env_dB = 20 * log10(X_env);

% get the initial and optimized models frequency-response:
H_init     = generateEqualizerCurve(fftSize/2, eqParamsInitial);
H_model    = generateEqualizerCurve(fftSize/2, eqParamsOptimized);
H_init_dB  = 20 * log10(H_init);
H_model_dB = 20 * log10(H_model);

%--------------------------------------------------------------------------
% from here: visualization

% plot spectrum and spectral envelope:
figure;
plot(binFreqs(plotRange), X_dB(plotRange), ... 
     binFreqs(plotRange), X_env_dB(plotRange));
 title('Spektrum und Hüllkurve via True Spectral Envelope');
 axis([0, max(binFreqs(plotRange)), -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');
 legend('Spektrum', 'Hüllkurve');
 
figure; 
plot(binFreqs(plotRange), X_env_dB(plotRange), ...
     binFreqs(plotRange), H_init_dB(plotRange), ...
     binFreqs(plotRange), H_model_dB(plotRange));
 title('Spektrale Hüllkurve und Modellfrequenzgang (nach Initialisierung und Optimierung)');
 axis([0, max(binFreqs(plotRange)), -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');
 legend('Hüllkurve', 'Modell (Initialisierung)' , 'Modell (optimiert)'); 
 

