clear all;

% assign the scale-factors for the parameters - these factors will be
% multiplied with the original parameters to achieve the desired
% formant-transformation:
formantFreqScale  = [1.00; 1.50; 1.00; 1.00; 1.00];
formantBwScale    = [1.00; 1.00; 1.00; 1.00; 1.00];
formantGainScale  = [1.00; 1.00; 1.00; 1.00; 1.00];
formantPhaseScale = [1.00; 1.00; 1.00; 1.00; 1.00];
normalizeGain     = 1; % if set to 1,  A will be scaled to compensate for 
                       % a bandwidth change (A_new *= aplha_new/alpha_old)

% assign the analysis parameters:
lpcOrder         = 10;         % model order, should be even
numStages        = lpcOrder/2; % number of parallel filter-stages
blockSize        = 8192;       % length of the block used for modeling
fftSize          = blockSize;  % FFT-size for visualization
decimationFactor = 4;          % the decimation factor
preEmph          = true;       % switches pre-emphasis on and off

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
% decomposition of the direct form allpole-model to the parallel form
% allpole-model and conversion to FOF-representation:

% expand the model transfer-function into (complex) partial fractions of
% the form R(i)/(1-P(i)*z^-1) such that the transfer-function can be
% expressed as H(z) = sum_{i=1}^N R(i)/(1-P(i)*z^-1)
[R,P,K] = residuez(1, errorCoeffs);

% combine complex conjugate one-pole-filters to real biquad-filters
% (actually they are not really biquads as the numerator is only linear in
% z):
S   = length(P)/2; % number of biquad stages
b0s = zeros(S,1);
b1s = zeros(S,1);
b2s = zeros(S,1);
a0s = zeros(S,1);
a1s = zeros(S,1);
a2s = zeros(S,1);
p   = 1;
for s=1:S
 b0s(s) = 2*real(R(p));              % b0 for stage s
 b1s(s) = -2*real(R(p)*conj(P(p)));  % b1 for stage s
 b2s(s) = 0;                         % b2 for stage s       
 
 a0s(s) = 1;                         % a0 for stage s
 a1s(s) = -2*real(P(p));             % a1 for stage s
 a2s(s) = abs(P(p))^2;               % a2 for stage s
 
 p = p+2; % increment for the poles (we skip every other pole as it is the
          % complex conjugate of the current pole)
end

% calculate the gain for each stage and scale the b-coefficients such that
% the b0s are normalized to unity:
gs  = b0s;
b0s = b0s./gs;
b1s = b1s./gs;
b2s = b2s./gs;

% convert the filter-coefficients to FOF-parameters:
[As, alphas, omegas, phis] = filterCoeffsToFofParams(gs, b1s, a1s, a2s);

% sort the formants by their frequency in ascending order:
parameters = [As, alphas, omegas, phis];
parameters = sortrows(parameters ,3);
As         = parameters(:,1);
alphas     = parameters(:,2);
omegas     = parameters(:,3);
phis       = parameters(:,4);

%--------------------------------------------------------------------------
% the actual formant-transformation:

% calculate the new (transformed) FOF-parameters:
As_new     = formantGainScale.*As;
alphas_new = formantBwScale.*alphas;
omegas_new = formantFreqScale.*omegas;
phis_new   = formantPhaseScale.*phis;
if( normalizeGain==1 )
 As_new = As_new.*(alphas_new./alphas);
elseif( normalizeGain == 2 ) 
 As_new = sqrt(As_new.^2.*(alphas_new./alphas));
elseif( normalizeGain == 3 )
 As_new = As_new.*sqrt(alphas_new./alphas);
end

% calculate the new (transformed) synthesis filter coefficients:
[gs_new, b1s_new, a1s_new, a2s_new] = fofParamsToFilterCoeffs(As_new, ... 
                                       alphas_new, omegas_new, phis_new);
a0s_new = ones(numStages,1);
b0s_new = ones(numStages,1);
b2s_new = zeros(numStages,1);

% apply the prediction-error-filter (which is the inverse model-filter) to
% whiten the input-signal:
whitenedSignalDec = filter(errorCoeffs, 1, inputSignalDec);
whitenedBlockDec  = filter(errorCoeffs, 1, windowedBlockDec);

% filter the whitened signal with the transformed synthesis filter:
transformedSignalDec = biquadBank(b0s_new, b1s_new, b2s_new, ...
                                  a0s_new, a1s_new, a2s_new, ...
                                  gs_new, ...
                                  whitenedSignalDec);
transformedBlockDec  = biquadBank(b0s_new, b1s_new, b2s_new, ...
                                  a0s_new, a1s_new, a2s_new, ...
                                  gs_new, ...
                                  whitenedBlockDec);        
                                                              
% apply inverse pre-emphasis filer:
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
H_new = zeros(fftSize/decimationFactor,1);
for s=1:numStages
 B = [b0s_new(s); b1s_new(s); b2s_new(s)];
 A = [a0s_new(s); a1s_new(s); a2s_new(s)];
 G = gs_new(s); 
 H_new = H_new + freqz(G*B, A, fftSize/decimationFactor, 'whole');
end
H_new = abs(H_new);
H_new = H_new./max(H_new);

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
                                                                 