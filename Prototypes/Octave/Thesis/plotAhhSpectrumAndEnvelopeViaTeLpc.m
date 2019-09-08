clear all;

% parameter settings:
blockSize = 8192;
fftSize   = blockSize;
lpcOrder  = 100;
preEmph   = true;

% read the input signal:
[inputSignal, sampleRate] = wavread('signals\Ahh.wav');

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
spectrum = abs(fft(windowedBlock));

% calculate lpc-coefficients:
predictionCoeffs = real(telpc(windowedBlock, lpcOrder));
predictionCoeffs = predictionCoeffs';

% convert into the prediction error filter:
errorCoeffs    = [-predictionCoeffs];
errorCoeffs(1) = -1;

% the impulse response of the prediction error filter is the same as it's
% coefficients (it's a FIR):
impResp                 = zeros(fftSize,1);
impResp(1:lpcOrder+1,1) = errorCoeffs;
  
% transfrom to the frequency-domain:
spectralEnvelope = abs(fft(impResp));

% invert the spectrum of the prediction error filter (the inverse
% prediction error filter is our model-filter):
spectralEnvelope = 1./spectralEnvelope;

% normalize gain:
G = (1/blockSize) * sum( abs(spectrum).^2 + abs(spectralEnvelope).^2  );
% spectralEnvelope = spectralEnvelope./max(spectralEnvelope);
% spectralEnvelope = spectralEnvelope.*normalizer;

% calculate the bin-frequencies
binFreqs = 0:(fftSize)-1;
binFreqs = binFreqs * sampleRate/fftSize;

% normalize spectrum and envelope such that the maximum value of the
% spectrum is 0 dB:
normalizer       = 1/max(spectrum);
spectrum         = normalizer*spectrum;
spectralEnvelope = normalizer*spectralEnvelope;

% convert spectrum and spectral envelope into dB:
dBSpectrum = 20 * log10(spectrum);
dBEnvelope = 20 * log10(spectralEnvelope);

% plot spectrum and spectral envelope:
figure;
plot(binFreqs(1:fftSize/4), dBSpectrum(1:fftSize/4), 'b', binFreqs(1:fftSize/4), dBEnvelope(1:fftSize/4), 'k');
 title('Spektrum und Hüllkurve via lineare Prädiktion');
 axis([0, max(binFreqs)/4, -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');
 legend('Spektrum', 'Hüllkurve');

