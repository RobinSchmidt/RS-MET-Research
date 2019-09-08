clear all;

% user parameter settings:

% load the input signal:
[inputSignal, sampleRate] = wavread('signals\Repraesentation.wav');

% the true envelope algorithm needs the fundamental frequency of the signal
% in order to adjust the cepstral order, so we need to specify this here:
fundamental  = 100;
% (automatic detection may be implemented later on a block-by-block basis
% but this is too expensive with the current pitch detection function due
% to the expensiveness of the xcorr-function):

% assign the scaling factor:
scaleFactor = 0.7;

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

% append some zeros at the beginning and at the end of the signals to make sure
% that the first samples and the last samples will not be underrepresented: 
padding1    = zeros( (blockSize/2), 1);
inputSignal = [padding1;  inputSignal; padding1];

% calculate the number of frames:
numFrames  = ceil(length(inputSignal)/hopSize);

% append some zeros at the end to make sure that the signal can be sliced
% into an integer number of frames:
desiredLength = numFrames*fftSize - (numFrames-1)*(fftSize-hopSize);
missingLength = desiredLength-length(inputSignal);
padding2      = zeros(missingLength,1);
inputSignal   = [inputSignal; padding2];

% generate the cos^2-window:
window = cosineSquaredWindow(blockSize);
 
%--------------------------------------------------------------------------
% the actual formant-scaling algorithm:

% allocate and initialize memory for the outputSignal:
outputSignal = zeros(length(inputSignal),1);

% allocate and initialize memory for the zero-padded frames:
paddedInputFrame = zeros(fftSize,1);

% loop through the frames
for m=1:numFrames
 
 disp(strcat('frame_', num2str(m), '_of_', num2str(numFrames)));
 
 % cut out the frame from the signal:
 inFrameStart  = (m-1) * hopSize + 1;
 inFrameEnd    = inFrameStart + blockSize - 1;
 outFrameStart = inFrameStart;
 outFrameEnd   = outFrameStart + fftSize - 1; 
 inputFrame    = window.*inputSignal(inFrameStart:inFrameEnd);

 % get the zero-padded frames by copying the actual samples into the first
 % part of the allocated vectors:
 paddedInputFrame(1:blockSize) = inputFrame;
  
 % get the spectrum of the frame:
 inputSpectrum  = fft( paddedInputFrame);
 
 % get the spectral envelope of the carrier and modulator spectrum:
 inputSpectralEnv = spectralEnvelopeViaTE(inputSpectrum, ...
                                          sampleRate, ...
                                          fundamental);
                                                                                
 % whiten spectrum:
 whitenedSpectrum = inputSpectrum./inputSpectralEnv;                          

 % transform the spectral envelope according to the warping map:
 outputSpectralEnv = inputSpectralEnv; % preliminary for test
 outputSpectralEnv = zeros(length(inputSpectralEnv),1);
 
 % apply the transformed spectral envelope to the frame:
 for k=1:fftSize/2+1
  readPosition = min( (k/scaleFactor), fftSize/2 );
  intPart      = floor(readPosition);
  fracPart     = readPosition-intPart; 
  if( intPart < 1 || (intPart+1) >= fftSize/2+1 )
   outputSpectralEnv(k) = 0;
  else
   % linear interpolation of the input-envelope:
   outputSpectralEnv(k) =   (1-fracPart)*inputSpectralEnv(intPart) ...
                          + fracPart*inputSpectralEnv(intPart+1);
  end
 end
 % symmetrize envelope:
 for k=2:(fftSize/2)
  q = k-2;
  outputSpectralEnv(fftSize-q) = outputSpectralEnv(k);  
 end
 outputSpectralEnv(fftSize/2+1) = inputSpectralEnv(fftSize/2+1);  

 % generate the spectrum of the output-frame:
 outputSpectrum = whitenedSpectrum.*outputSpectralEnv;
 
 % transform to the time domain:
 outputFrame = real(ifft(outputSpectrum));
  
 % overlap/add:
 outputSignal(outFrameStart:outFrameEnd) =  outputSignal(outFrameStart:outFrameEnd) + outputFrame;
 
end

% calculate the start and end sample position for the output-signal to make
% its length fit the original length of the input-signal (if this option is
% chosen):
if ( cutToOriginalLength )
 outStart = length(padding1) + 1;
 outEnd   = length(outputSignal) - length(padding1) - length(padding2);
else
 outStart = 1;
 outEnd   = length(outputSignal);
end

% cut the signal to its original length (if this is option chosen):
outputSignal = outputSignal(outStart:outEnd);

% play:
soundsc(outputSignal, sampleRate);


