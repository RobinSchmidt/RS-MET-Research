clear all;

% assign the scale-factors for the parameters - these factors will be
% multiplied with the original parameters to achieve the desired
% formant-transformation:
formantFreqScale = [1.00; 1.50; 1.00; 1.00; 1.00];
formantBwScale   = [1.00; 1.00; 1.00; 1.00; 1.00];

% assign the analysis parameters:
lpcOrder         = 10;        % model order, should be even
blockSize        = 8192;      % length of the block used for modeling
fftSize          = blockSize; % FFT-size for visualization
decimationFactor = 4;         % the decimation factor
preEmph          = true;      % switches pre-emphasis on and off

% read the input signal:
[inputSignal, sampleRate] = wavread('signals\Ahh.wav');

% take only a chunk (from somwhere in the middle):
signalBlock = inputSignal(10000:10000+blockSize-1);

%--------------------------------------------------------------------------
% decimation and pre-emphasis:

% decimate the signal and the signal-block:
inputSignalDec = decimate(inputSignal, decimationFactor, 'FIR');
signalBlockDec = decimate(signalBlock, decimationFactor, 'FIR');
sampleRateDec  = sampleRate/decimationFactor;

% create the reisdual-signal by interpolating the decimated signal
% (yielding the anti-alias-filtered input-signal) and subtracting it from
% the original signal:
inputSignalInt     = interp(inputSignalDec, decimationFactor);
inputSignalInt     = inputSignalInt(1:length(inputSignal));
decimationResidual = inputSignal - inputSignalInt;

% apply a hanning window to the decimated block:
windowedBlockDec = signalBlockDec.*hann(length(signalBlockDec));

% apply optimal pre-emphasis to the decimated signal:
preEmphCoeff = 0;
if(preEmph)
 % get the optimal pre-emphasis coefficient (the optimal one-step
 % predictor):
 coeffs       = real(lpc(windowedBlockDec, 1));
 preEmphCoeff = coeffs(2);
 % apply pre-emphasis to the signal:
 inputSignalDec   = filter(coeffs, 1, inputSignalDec);
 windowedBlockDec = filter(coeffs, 1, windowedBlockDec);
end

%--------------------------------------------------------------------------
% AR-modeling:

% calculate lpc-coefficients:
predictionCoeffs = real(telpc(windowedBlockDec, lpcOrder, ... 
                              50, 300, sampleRateDec));
%predictionCoeffs = real(lpc(windowedBlockDec, lpcOrder));
predictionCoeffs = predictionCoeffs';

% convert into the prediction error filter:
errorCoeffs    = [-predictionCoeffs];
errorCoeffs(1) = -1;

%--------------------------------------------------------------------------
% decomposition of the direct form allpole-model to the serial 
% allpole-model and conversion to formant frequencies and bandwidths:

% convert coefficients to poles:
poles      = roots(errorCoeffs);
poleRadii  = abs(poles);
poleAngles = angle(poles);

% calculate the formant frequencies and bandwidths:
formantFreqs = zeros(length(poleAngles)/2,1);
formantBws   = zeros(length(poleRadii)/2,1);
p            = 1; 
for s=1:length(formantFreqs)
 formantFreqs(s) = poleAngles(p) * sampleRateDec / (2*pi);
 formantBws(s)   = sampleRateDec * (-log(poleRadii(p)) / pi);
 p = p + 2; 
end

% sort the formants by their frequency in ascending order:
formants     = [formantFreqs, formantBws];
formants     = sortrows(formants ,1);
formantFreqs = formants (:,1);
formantBws   = formants (:,2);

%--------------------------------------------------------------------------
% the actual formant-transformation:

% calculate the new (transformed) formant-frequencies and -bandwidths:
newFormantFreqs = formantFreqScale.*formantFreqs;
newFormantBws   = formantBwScale.*formantBws;

% calculate the new poles:
newPoles      = poles;
newPoleRadii  = poleRadii;
newPoleAngles = poleAngles;
for p=1:length(formantFreqScale)
 poleAngle            = 2*pi* newFormantFreqs(p) / sampleRateDec;
 poleRadius           = exp( - newFormantBws(p)*pi/sampleRateDec ); 
 newPoleRadii(2*p-1)  = poleRadius;
 newPoleRadii(2*p)    = poleRadius;  
 newPoleAngles(2*p-1) = poleAngle;
 newPoleAngles(2*p)   = -poleAngle;  
end
newPoles =     newPoleRadii.*cos(newPoleAngles) + ...
           i * newPoleRadii.*sin(newPoleAngles);

% convert the new poles into a direct form filter:
newCoeffs = real(poly(newPoles))';

% apply the prediction-error-filter (which is the inverse model-filter) to
% whiten the input-signal:
whitenedSignalDec = filter(errorCoeffs, 1, inputSignalDec);
whitenedBlockDec  = filter(errorCoeffs, 1, windowedBlockDec);

% apply the modified filter to the signal:
transformedSignalDec = filter(1, newCoeffs, whitenedSignalDec);
transformedBlockDec  = filter(1, newCoeffs, whitenedBlockDec);

% apply inverse pre-emphasis filter:
if(preEmph)
 inputSignalDec       = filter(1, [1; preEmphCoeff], inputSignalDec);
 windowedBlockDec     = filter(1, [1; preEmphCoeff], windowedBlockDec); 
 transformedSignalDec = filter(1, [1; preEmphCoeff], transformedSignalDec);
 transformedBlockDec  = filter(1, [1; preEmphCoeff], transformedBlockDec); 
end

% interpolate the transformed signal to the original sample-rate and add the
% decimation residual:
transformedSignal = interp(transformedSignalDec, decimationFactor);
transformedSignal = transformedSignal(1:length(inputSignal));
transformedSignal = transformedSignal + decimationResidual;  

%--------------------------------------------------------------------------
% visualization of the results:

% calculate spectrum of the input signal block:
X = abs(fft(windowedBlockDec));
X = X./max(X);

% calculate spectrum of the transformed signal block:
Y = abs(fft(transformedBlockDec));
Y = Y./max(Y);

% calculate the frequency-response of the model:
H = abs(freqz(1, errorCoeffs, fftSize/decimationFactor, 'whole'));
H = H/max(H);

% calculate the frequency-response of the transformed model:
H_new = abs(freqz(1, newCoeffs, fftSize/decimationFactor, 'whole'));
H_new = H_new/max(H_new);

% convert spectra to decibels:
X_dB     = 20 * log10(X);
H_dB     = 20 * log10(H);
Y_dB     = 20 * log10(Y);
H_new_dB = 20 * log10(H_new);

% calculate the values for the frequency-axis
numBins  = length(X);
Omegas   = (0:1:(numBins-1))';
Omegas   = 2*pi*Omegas/numBins;
binFreqs = (0:1:(numBins-1))';
binFreqs = sampleRateDec*binFreqs/numBins;

% plot:
figure;
plot(binFreqs(1:length(X)/2), Y_dB(1:length(X)/2), 'b', ...
     binFreqs(1:length(X)/2), H_new_dB(1:length(X)/2), 'k' );
 axis([0 binFreqs(length(X)/2) -60 6]);
 grid on;
 
figure;
plot(binFreqs(1:length(X)/2), H_dB(1:length(X)/2), 'k', ...
     binFreqs(1:length(X)/2), H_new_dB(1:length(X)/2), 'k' );
 axis([0 binFreqs(length(X)/2) -60 6]);
 grid on; 

% play:
soundsc(transformedSignal, sampleRate);