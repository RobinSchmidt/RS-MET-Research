clear all;

% parameter settings:
blockSize = 8192;
fftSize   = blockSize;
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
spectrum = spectrum(1:fftSize/2);

% normalize:
normalizer = 1/max(spectrum);
spectrum   = spectrum.*normalizer;

% calculate the bin-frequencies
binFreqs = 0:(fftSize)-1;
binFreqs = binFreqs * sampleRate/fftSize;

% convert spectrum into dB:
dBSpectrum = 20 * log10(spectrum);

% plot spectrum and spectral envelope:
plot(binFreqs(1:fftSize/4), dBSpectrum(1:fftSize/4), 'k');
 title('Spektrum eines männlichen "Ahh"');
 axis([0, max(binFreqs)/4, -60.0, 6.0]);
 grid on;
 xlabel('Frequenz in Hz');
 ylabel('Amplitude in dB');


