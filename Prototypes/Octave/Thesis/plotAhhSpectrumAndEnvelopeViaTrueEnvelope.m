clear all;

% parameter settings:
blockSize      = 8192;
fftSize        = blockSize;
preEmph        = true;
cepstralWindow = 1;   % 0 for rectangular, 1 for hamming window

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

% estimate the spectral envelope:
[X_env, C, P_c] = spectralEnvelopeViaTE(X, sampleRate, fundamental, 1.0, 0.8, cepstralWindow, 1.0);

% convert spectrum and spectral envelope to decibels:
X_dB     = 20 * log10(X);
X_env_dB = 20 * log10(X_env);

% plot spectrum and spectral envelope:
plot(binFreqs(1:fftSize/4), X_dB(1:fftSize/4), binFreqs(1:fftSize/4), X_env_dB(1:fftSize/4), 'k');
 title('Spektrum und Hüllkurve via True Spectral Envelope');
 axis([0, max(binFreqs)/4, -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');
 legend('Spektrum', 'Hüllkurve');

