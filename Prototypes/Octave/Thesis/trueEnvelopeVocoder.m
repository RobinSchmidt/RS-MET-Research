clear all;

% user parameter settings:

% choose the signals to be used as modulator and carrier (they must have
% the same sample-rate):
[modulator]           = audioread('signals\Repraesentation.wav');
[carrier, sampleRate] = audioread('signals\Chor.wav');
% todo: revert to MatLab's wavread - implement wavread for Octave as wrapper
% for audioread - similar for wavwrite

% the true envelope algorithm needs the fundamental frequency of the signal
% in order to adjust the cepstral order, so we need to specify this here:
carrierFundamental   = 200;
modulatorFundamental = 100;
% (automatic detection may be implemented later on a block-by-block basis
% but this is too expensive with the current pitch detection function due
% to the expensiveness of the xcorr-function):

% choose, if (and how much) the carrier should be whitened (values should
% be between 0...1):
carrierWhitening = 1.0;

% choose, if the vocoder output-signal should be cut to the original length
% of the modulator signal (there are some zeros appended before the vocoder
% algorithm starts to take care of some windowing/overlapping side effects):
cutToOriginalLength = true;

% choose the block-related parameters:
blockSize = 2048;
hopSize   = blockSize/2;
fftSize   = 2*blockSize;

%--------------------------------------------------------------------------
% pre-calculations:

% determine the number of channels (i.e. stereo or mono) of the carrier and
% the modulator:
[dummy, numChannelsModulator] = size(modulator);
[dummy, numChannelsCarrier]   = size(carrier);

% decompose the channels of the signals (which are possibly in
% stereo-format):
carrierL = carrier(:,1);
if( numChannelsCarrier == 2 )
 carrierR = carrier(:,2);
else
 carrierR = carrierL; 
end
modulatorL = modulator(:,1);
if( numChannelsModulator == 2 )
 modulatorR = modulator(:,2);
else
 modulatorR = modulatorL; 
end

% lengthen the carrier-signal until it is as long as or longer than the
% modulator-signal by looping it:
while( length(carrierL) < length(modulatorL) )
 carrierL = [carrierL; carrierL]; 
 carrierR = [carrierR; carrierR];   
end

% now the carrier is long enough but maybe too long, so shorten it to
% exactly fit the length of the modulator:
carrierL = carrierL(1:length(modulatorL));
carrierR = carrierR(1:length(modulatorL));

% append some zeros at the beginning and at the end of the signals to make sure
% that the first samples and the last samples will not be underrepresented: 
padding1   = zeros( (blockSize/2), 1);
carrierL   = [padding1; carrierL; padding1];
carrierR   = [padding1; carrierR; padding1]; 
modulatorL = [padding1; modulatorL; padding1];
modulatorR = [padding1; modulatorR; padding1];

% calculate the number of frames:
numFrames  = ceil(length(carrierL)/hopSize);

% append some zeros at the end to make sure that the signal can be sliced
% into an integer number of frames:
desiredLength = numFrames*fftSize - (numFrames-1)*(fftSize-hopSize);
missingLength = desiredLength-length(carrierL);
padding2      = zeros(missingLength,1);
carrierL      = [carrierL; padding2];
carrierR      = [carrierR; padding2]; 
modulatorL    = [modulatorL; padding2];
modulatorR    = [modulatorR; padding2];

% generate the cos^2-window:
window = cosineSquaredWindow(blockSize);

%--------------------------------------------------------------------------
% the actual vocoder algorithm:

% allocate and initialize memory for the outputSignal:
vocoL = zeros(length(carrierL),1);
vocoR = zeros(length(carrierL),1);

% allocate and initialize memory for the zero-padded frames:
paddedCarrierLFrame   = zeros(fftSize,1);
paddedCarrierRFrame   = zeros(fftSize,1);
paddedModulatorLFrame = zeros(fftSize,1);
paddedModulatorRFrame = zeros(fftSize,1);

% loop through the frames
for m=1:numFrames
 
 disp(strcat('frame_', num2str(m), '_of_', num2str(numFrames)));
 
 % cut out the frames from the carrier and the modulator:
 inFrameStart    = (m-1) * hopSize + 1;
 inFrameEnd      = inFrameStart + blockSize - 1;
 outFrameStart   = inFrameStart;
 outFrameEnd     = outFrameStart + fftSize - 1;
 carrierLFrame   = window.*carrierL(inFrameStart:inFrameEnd);
 carrierRFrame   = window.*carrierR(inFrameStart:inFrameEnd); 
 modulatorLFrame = window.*modulatorL(inFrameStart:inFrameEnd);
 modulatorRFrame = window.*modulatorR(inFrameStart:inFrameEnd); 
 
 % get the zero-padded frames by copying the actual samples into the first
 % part of the allocated vectors:
 paddedCarrierLFrame(1:blockSize)   = carrierLFrame;
 paddedCarrierRFrame(1:blockSize)   = carrierRFrame;
 paddedModulatorLFrame(1:blockSize) = modulatorLFrame;  
 paddedModulatorRFrame(1:blockSize) = modulatorRFrame;   
 
 % get the spectra of the carrier and modulator frame:
 carrierLSpectrum  = fft(paddedCarrierLFrame);
 if( numChannelsCarrier == 2 )
  carrierRSpectrum = fft(paddedCarrierRFrame); 
 else % avoid second FFT, if carrier is mono
  carrierRSpectrum = carrierLSpectrum;
 end
 modulatorLSpectrum = fft(paddedModulatorLFrame); 
 if( numChannelsModulator == 2 )
  modulatorRSpectrum = fft(paddedModulatorRFrame);   
 else % avoid second FFT, if modulator is mono
  modulatorRSpectrum = modulatorLSpectrum;  
 end
 
 % get the spectral envelope of the carrier and modulator spectrum:
 carrierLSpectralEnv = spectralEnvelopeViaTE(carrierLSpectrum, ...
  sampleRate, ...
  carrierFundamental);    
 if( numChannelsCarrier == 2 )
  carrierRSpectralEnv = spectralEnvelopeViaTE(carrierRSpectrum, ...
   sampleRate, ...
   carrierFundamental); 
 else % avoid second True-Envelope Algorithm, if carrier is mono
  carrierRSpectralEnv = carrierLSpectralEnv;
 end
 
 modulatorLSpectralEnv = spectralEnvelopeViaTE(modulatorLSpectrum, ...
  sampleRate, ...
  modulatorFundamental);
 if( numChannelsModulator == 2 )
  modulatorRSpectralEnv = spectralEnvelopeViaTE(modulatorRSpectrum, ...
   sampleRate, ...
   modulatorFundamental); 
 else % avoid second True-Envelope Algorithm, if modulator is mono
  modulatorRSpectralEnv = modulatorLSpectralEnv;
 end
 
 % whiten carrier spectrum:
 c = carrierWhitening;
 whitenedCarrierLSpectrum = (1-c)*carrierLSpectrum + ...
  c*carrierLSpectrum./carrierLSpectralEnv;                          
 whitenedCarrierRSpectrum = (1-c)*carrierRSpectrum + ...
  c*carrierRSpectrum./carrierRSpectralEnv;
 
 % produce the vocoder-output by multiplying the (whitened) carrier
 % spectrum by the modulators spectral envelope:
 vocoLFrame = real(ifft(modulatorLSpectralEnv.*whitenedCarrierLSpectrum));
 vocoRFrame = real(ifft(modulatorRSpectralEnv.*whitenedCarrierRSpectrum)); 
 
 % overlap/add:
 vocoL(outFrameStart:outFrameEnd) =   vocoL(outFrameStart:outFrameEnd) ...
  + vocoLFrame;
 vocoR(outFrameStart:outFrameEnd) =   vocoR(outFrameStart:outFrameEnd) ...
  + vocoRFrame; 
 
end

% normalize the vocoder-signal:
normalizer = max(max(vocoL), max(vocoR));
vocoL      = normalizer*vocoL;
vocoR      = normalizer*vocoR;

% calculate the start and end sample position for the output-signal to make
% its length fit the original length of the modulator (if this option is
% chosen):
if ( cutToOriginalLength )
 outStart = length(padding1) + 1;
 outEnd   = length(vocoL) - length(padding1) - length(padding2);
else
 outStart = 1;
 outEnd   = length(vocoL);
end

% combine the channels to a stereo signal:
out = [vocoL(outStart:outEnd), vocoR(outStart:outEnd)];

% write output-signal to a wave-file and play it:
%wavwrite(out, sampleRate, 16, 'signals\vocoderOutput.wav');
audiowrite('signals\vocoderOutput.wav', out, sampleRate)
soundsc(out, sampleRate);





