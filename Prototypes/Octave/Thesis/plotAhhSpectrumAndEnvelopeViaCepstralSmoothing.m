clear all;

% parameter settings:
blockSize     = 8192;
fftSize       = blockSize;
preEmph       = true;

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

% normalize:
normalizer = 1/max(spectrum);
spectrum   = spectrum.*normalizer;

% take the logarithm of the magnitude-spectrum:
logSpectrum = log(spectrum);

% do inverse dft to get the cepstrum:
cepstrum = ifft(logSpectrum);

% estimate the fundamental frequency of the input signal (in Hz):
fundamental = estimateFundamental(windowedBlock, sampleRate, 50, 1000);

% apply "liftering" in the cepstral domain:
cepstralOrder = 1.0 * ceil(sampleRate / (2*fundamental) );

% apply "liftering" in the cepstral domain:
lifteredCepstrum   = cepstrum;
lifteredCepstrum( cepstralOrder:(fftSize-cepstralOrder+2) ) = 0;

% calculate the spectral envelope:
spectralEnvelope = exp(fft(lifteredCepstrum));

% calculate the bin-frequencies
binFreqs = 0:(fftSize)-1;
binFreqs = binFreqs * sampleRate/fftSize;

% convert spectrum and spectral envelope into dB:
dBSpectrum = 20 * log10(spectrum);
dBEnvelope = 20 * log10(spectralEnvelope);

% plot spectrum and spectral envelope:
plot(binFreqs(1:fftSize/4), dBSpectrum(1:fftSize/4), 'b', binFreqs(1:fftSize/4), dBEnvelope(1:fftSize/4), 'k');
 title('Spektrum und Hüllkurve via Cepstral Smoothing');
 axis([0, max(binFreqs)/4, -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');
 legend('Spektrum', 'Hüllkurve');

