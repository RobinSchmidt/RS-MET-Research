function parameters = optimizeEQParametersViaSCG(initialParameters, ...
                                                 X_env, w, ...
                                                 showProgress, ...
                                                 sampleRate, ...
                                                 plotRange);
                                                
% This function optimizes a set of initial equalizer-parameters in order to
% fit the frequency response to a given spectral envelope. It uses the
% scaled conjugate gradient algorithm for the parameter update.
%
% usage: 
%  parameters = optimizeEQParametersViaSCG(initialParameters, ...
%                                          X_env, w);
%
% input-variables:
%  -initialParameters: the vector of initial equalizer-parameters 
%    [G; g; OmegaC; gamma]
%  -X_env: the spectral envelope of the input signal (from 0-pi - that is:
%    the redundant bins are already cut off)
%  -w: the error-weighting function (again, without the redundant bins)
%  -showProgress: this is the number of iterations after which an
%    intermediate result wil be shown. if zero is passed here, no
%    intermediate results will be shown.
%  -sampleRate: needed for the frequency-axis scale when showProgress
%    is active
%
% output-variables:
%  -parameters: the optimized equalizer-parameters after the algorithm has
%    converged

%--------------------------------------------------------------------------

if( nargin<4 )
 showProgress = false; 
end

% assign some internal constants for the SCG-algorithm:
betaFormula = 1;      % uses the Polak-Ribiere form
epsilon     = 1.e-3;  % step-size for second order estimation
lambda_min  = 1.e-7;  % minimum value for lambda
lambda_max  = 1.e+8;  % maximum value for lambda
withReset   = true;

% get the number of (non-redundant) bins and the number of equalize-stages:
numBins   = length(X_env);
numStages = round((length(initialParameters)-1)/3);

% check the error-weighting function for the minimum and maximum bin with
% nonzero weight:
currentBin = 1;
isZero     = true;
while( isZero )
 if( w(currentBin) == 0 )
  isZero = true;  
 else
  isZero = false;  
 end
 currentBin = currentBin + 1; 
end
minBin = currentBin - 1;

currentBin = length(w);
isZero     = true;
while( isZero )
 if( w(currentBin) == 0 )
  isZero = true;  
 else
  isZero = false;  
 end
 currentBin = currentBin - 1; 
end
maxBin = currentBin + 1;


% initialize converged-flag and iteration-counter:
converged = false;
iteration = 0;

% do the initial step of the algorithm:
parameters = updateParametersViaSCG(1, initialParameters, X_env, w, ... 
                                    minBin, maxBin, betaFormula, ... 
                                    epsilon, lambda_min, lambda_max, ... 
                                    withReset);
iteration  = iteration + 1; % this was already the first iteration

if( showProgress ~= 0 )
 % generate the equalizers initial frequency response curve:
 H_init = generateEqualizerCurve(numBins, initialParameters); 
 
 % assign a figure-window for the plot inside the loop:
 figure;   
 
 % calculate bin-frequencies:
 fftSize  = 2*length(X_env);
 binFreqs = 0:(plotRange)-1;
 binFreqs = binFreqs * sampleRate/fftSize;
 
 % convert the spectral envelope, the eq-curve and the error-weights
 % to decibels:
 X_env_dB  = 20 * log10(X_env+eps);
 H_init_dB = 20 * log10(H_init+eps);
 w_dB      = 20 * log10(w+eps);
 w_dB      = max(w_dB, min(X_env_dB));
 
 
 % plot the spectral envelope and the initial eq-curve:
 plot(binFreqs, X_env_dB(1:plotRange), ... 
      binFreqs, H_init_dB(1:plotRange), ... 
      binFreqs, w_dB(1:plotRange));  
 grid on;
 pause;
end

while (~converged && iteration<=2000)
 
 %errorOld = calculateErrorFunction(parameters, X_env, w);
 
 % do the update step:
 parameters = updateParametersViaSCG(0, parameters, X_env, w, ... 
                                     minBin, maxBin, betaFormula, ... 
                                     epsilon, lambda_min, lambda_max, ... 
                                     withReset);
                                    
 % force all parameters to be > 0
 for( k=1:length(parameters) )
  if( parameters(k) < 0.00000001 )
   parameters(k) = 0.00000001;  
  end  
 end
 
 %errorNew = calculateErrorFunction(parameters, X_env, w);
 
 
 % increment the iteration counter:
 iteration = iteration + 1;                                    
                      
 % calculate and decompose the current gradient in order to check the
 % convergence-criterion:
 gradient   = calculateGradient(parameters, X_env, w); 
 dE_dG      = gradient(1);
 dE_dg      = gradient(2:(numStages+1));
 dE_dOmegaC = gradient((numStages+2):(2*numStages+1));
 dE_dgamma  = gradient((2*numStages+2):(3*numStages+1)); 
 
 % check convergence criterion:
 converged = abs(dE_dG)<0.001 && max(abs(dE_dg))<0.001 && ... 
             max(abs(dE_dOmegaC))<0.01 && max(abs(dE_dgamma))<0.01;
%  converged = abs(dE_dG)<0.001 && max(abs(dE_dg))<0.001 && ... 
%              max(abs(dE_dOmegaC))<0.004 && max(abs(dE_dgamma))<0.01;

 % show progress from time to time:
 if( (showProgress~=0) && mod(iteration,showProgress)==0 )
  
  % generate the current equalizers frequency response curve:
  H = generateEqualizerCurve(numBins, parameters);
  
  % convert to decibels:
  H_dB = 20 * log10(H);  
  
  % plot the curves:
  plot(binFreqs, X_env_dB(1:plotRange), ... 
       binFreqs, H_init_dB(1:plotRange), ... 
       binFreqs, H_dB(1:plotRange), ... 
       binFreqs, w_dB(1:plotRange)); 
  grid on;
  
  disp(gradient);
  
  pause;
 end
  
end



