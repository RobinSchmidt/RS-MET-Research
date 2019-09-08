function [X_env, C, P_c] = spectralEnvelopeViaTE(X, f_s, f_0, Delta, c, ...
                                                 wnd, smooth);

% This function estimates the spectral envelope via the True Spectral
% Envelope Algorithm as described in the paper "Efficient Spectral Envelope
% Estimation and its Application to Pitch Shifting and Envelope
% Preservation" by A. Robel and X. Rodet. The algorithm is implemented in
% its improved form with hamming-windowing and adaptive stepsize-control
% but without the downsampling optimization. But in contrast to the paper,
% the log-spectrum is replaced by a dB-spectrum (which differs from the
% log-spectrum only by a scaling factor - that is, we use the logarithm to
% the base 10 and multiply by 20 instead of just taking the natural 
% logarithm). Using the dB-representation instead of the natural logarithm
% makes the check of the convergence-criterion more straightforward.
%
% usage: 
%  [X_env, C, P_c] = spectralEnvelopeViaTE(X, f_s, f_0, Delta, c, ...
%                                          wnd, smooth);
%
% input-variables:
%  -X: FFT-spectrum to be enveloped
%  -f_s: sample-rate of the signal
%  -f_0: fundamental frequency of the signal
%  -Delta: maximum dB-value which is the envelope allowed to be below the
%    spectrum
%  -c: an exponent between 0...1 which controls the stepsize lambda - for c=0
%    the stepsize always will be lambda=1.0, for c=1 it will be lambda=E/E_in, 
%    in general: lambda = (E/E_in)^c with E and E_in being the total and
%    in-band energies of the difference-cepstrum between successive
%    iterations
%  -wnd: an index for the window-function to be used for the cepstral
%    liftering (0: rectangular, 1: hamming)
%  -smooth: a number between 0...1 to scale the cepstral order - values < 1
%    will give smoother envelopes, =1 will use the optimum order, for 
%    values > 1, individual partials will be resolved
%
% output-variables:
%  -X_env: the spectral envelope
%  -C: the cepstrum corresponding to X_env
%  -P_c: the cepstral order which has been used

%--------------------------------------------------------------------------

if( nargin<7 )
 smooth = 1; % no systematic smoothing by default
end
if( nargin<6 )
 wnd = 1;    % hamming window by default
end
if( nargin<5 )
 c = 0.8;    % default exponent for lambda
end
if( nargin<4 )
 Delta = 1;  % maximum allowed downward-deviation of the envelope at the 
             % spectral-peaks 
end

% get the fft-size:
fftSize = length(X);

% get the dB-representation of the magnitude-spectrum:
dBSpectrum  = 20 * log10(abs(X)+eps); % +eps to prevent log-of-zero

% do inverse dft to get the cepstrum:
cepstrum = ifft(X);

% choose maximum cepstral order acording to eq.(6) in the paper
% by Roebel/Rodet (with additional smoothing factor):
P_c = ceil( smooth * f_s / (2*f_0) );

% generate the cepstral windows (this must take into account the
% dft-symmetry):
switch( wnd )
 
 case 0 % generate the rectangular cepstral window
  W = ones(length(cepstrum),1);
  W( P_c:(fftSize-P_c+2) ) = 0;  
  
 case 1 % generate the hamming cepstral window
  W             = zeros(length(cepstrum),1);
  windowLength  = ceil(1.66*P_c);
  tempWindow    = hamming(2*windowLength);  
    % length of tempWindow is now assured to be even
  tempWindow    = tempWindow(length(tempWindow)/2+1:length(tempWindow)); 
    % we use the "decaying" part only
  R             = length(cepstrum);
  W(1)          = 1;
  for r=0:windowLength-1
   W(r+2) = tempWindow(r+1);
   W(R-r) = tempWindow(r+1);
  end 
  clear tempWindow;
  
 otherwise % rectangular window, when 'wnd' is not a valid index
  W = ones(length(cepstrum),1);
  W( P_c:(fftSize-P_c+2) ) = 0;  
  
end

%--------------------------------------------------------------------------
% from here: the actual True Envelope Algorithm:

V         = -inf * ones(fftSize,1);
A         = dBSpectrum;
C         = zeros(fftSize,1);   
converged = false;
iteration = 0;
while ~converged
 
 % choose the maximum (element wise):
 A = max(A, V);  

 % transform the target spectrum to the cepstral domain:
 C_dash = real(ifft(A));
 
 % calculate the difference between the previous and the current 
 % cepstral representation of the envelope:
 D = C_dash - C;
  
 % calculate in-band and total energies of the cepstral change
 % between the current and the previous iteration:
 E_tot = sum( abs(D).^2 );
 E_in  = sum( abs(W.*D).^2 );
 
 % calculate the optimal stepsize:
 if( iteration == 0)
  lambda = 1;
 else
  lambda = (E_tot / E_in)^c;    
 end;
 
 % do the update-step for C:
 C = C + lambda * W.*(D);
 
 % transfrom back to the log-spectral domain:
 V = real(fft(C)); 
 
 % check the convergence criterion:
 if( max(dBSpectrum - V) <= Delta || iteration > 500)
  converged = true; 
 end 
 
 iteration = iteration + 1;
 
end

% generate the output-variables:
C     = real(C);
X_env = 10.^(V/20);

