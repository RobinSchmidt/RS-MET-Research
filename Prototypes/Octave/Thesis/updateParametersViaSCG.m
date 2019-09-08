function newParameters = updateParametersViaSCG(init, parameters, X, ... 
                          w, minBin, maxBin, betaFormula, epsilon, ...
                          lambda_min, lambda_max, withReset)
                         
% This function performs one step of the scaled conjugate gradient
% algorithm to update an eq-parameter vector. It has some persistent
% variables to maintain their values between succesive function-calls
% (corrsesponding to successive optimization steps). To reset or initialize
% the algorithm, it should be called with "true" as the first 
% input-variable.
%
% usage: 
% newParameters = updateParametersViaSCG(init, parameters, X, ... 
%                  w, minBin, maxBin, betaFormula, epsilon, ...
%                  lambda_min, lambda_max, withReset)
%
% the function is assumed to be called inside a loop like this:
% converged  = false;
% parameters = updateParametersViaSCG(1, parameters, X, ...);
% while( ~converged )
%  parameters = updateParametersViaSCG(0, parameters, X, ...);
%  converged  = checkConvergenceCriterion(parameters);
% end
%
% input-variables:
%  -init: a flag that causes an initialization of the search direction as 
%    the negative gradient - should be true for the first call
%  -parameters: the current parameter vector
%  -X: the spectral envelope to be fitted by the equalizer frequency
%    response (from 0-pi - that is: the redundant bins are already cut off)
%  -w: the error-weighting function (again, from 0-pi)
%  -minBin: lowest nonzero bin of the error-weighting function (default=1)
%  -maxBin: highest nonzero bin of the error-weighting function 
%    (default = length(X))
%  -betaFormula: chooses, which formula should be used for the calculation
%    of beta 1: Polak-Ribiere (default and recommended), 2: Hestenes-Stiefel, 
%    3: Fletcher-Reeves, 4: original
%  -epsilon: stepsize for the estimation of the matrix-vector-product between
%    the Hessian and the current direction (default = 0.00005)
%  -lambda_min: floor-value for lambda - the lowest value, which lambda can
%    take on
%  -lambda_max:
%  -withReset: if set to true (default=false), the direction will be
%    periodically reset to the negative gradient after each D iterations 
%    (with D the dimensionality of 'parameters')
%
% output-variables:
%  -newParameters: the updated parameter-vector

%--------------------------------------------------------------------------                         
     
% declare the persistent variables (they are remembered between
% function-calls just like static variables in C):
persistent lambda;
persistent gradient;
persistent gradient_old;
persistent direction;
persistent direction_old;
persistent H_times_d;
persistent iteration;

persistent alpha beta delta Delta;
             
% the optional variables are pre-asiigned to default-values:
if(nargin < 4)  
 w = ones(length(X),1); % uses a unity error weight-function
end
if(nargin < 5)  
 minBin = 1;            % lowest nonzero bin in the error weight-function
end
if(nargin < 6)  
 maxBin = length(X);    % highest nonzero bin in the error weight-function
end
if(nargin < 7)  
 betaFormula = 1;       % Polak-Ribiere form by default
end
if(nargin < 8)  
 epsilon = 0.00005;     % stepsize for second order estimation
end
if(nargin < 9)  
 lambda_min = 1.e-6;    % minimum value for the lambda-parameter 
end
if(nargin < 10)  
 lambda_max = 1.e+8;    % maximum value for the lambda-parameter
end
if(nargin < 11)
 withReset = false; 
end

% initialize the algorithm, if the corresponding flag-parameter is true:
if( init==true )
 lambda        = 1;
 gradient      = calculateGradient(parameters, X, w, minBin, maxBin);
 gradient_old  = zeros(length(gradient),1);
 direction     = -gradient;
 direction_old = zeros(length(direction),1);
 H_times_d     = 1;
 iteration     = 0;
end

%--------------------------------------------------------------------------
% from here: the actual parameter update step of the scaled conjugate
% gradient algorithm: 

% increment the iteration counter:
iteration = iteration+1;

% evaluate the error-function:
error = calculateErrorFunction(parameters, X, w);

% estimate the product between the Hessian matrix and the current
% search-direction (remebering the old value):
scaledEpsilon = epsilon/norm(direction);
H_times_d_old = H_times_d;
H_times_d = (   calculateGradient(parameters+scaledEpsilon*direction, ... 
                                  X, w, minBin, maxBin) ...
              - calculateGradient(parameters, X, w, minBin, maxBin) ) ...
             / scaledEpsilon;

% choose the stepsize that minimizes the quadratic approximation of the
% error-function along the current direction (taking into acount the
% scaling-factor lambda):
delta = (direction' * H_times_d) + lambda * norm(direction)^2;
if( delta < 0 )
 disp('delta<0');
 lambda = 2*(lambda - delta/norm(direction)^2);
 delta  = -(direction' * H_times_d);
end
alpha = - (direction'*gradient) / delta;

% calculate the actual error (and its quadratic approximation) at the
% position which would result from the update step:
errorHere  = error;
errorNext  = calculateErrorFunction(parameters + alpha*direction, X, w);
%errorNextQ = errorHere + alpha*direction'*gradient + 0.5*alpha^2*direction'*H_times_d;

% calculate the comparison-parameter between the actual error-function
% and it's quadratic approximation:
%Delta = (errorHere-errorNext) / (errorHere-errorNextQ);
Delta = -2*(errorHere-errorNext) / (alpha*direction'*gradient);

% update the parameter-vector, but only if the error will really decrease:
% if( (errorHere - errorNext) >= 0 )
if( Delta >= 0 )
 
 % update the parameters:
 newParameters = parameters + alpha * direction;

 % decrease lambda, if the approximation is good, increase lambda, if the
 % approximation is bad:
 if( Delta > 0.75 )
  lambda = 0.5 * lambda;
  lambda = max(lambda, lambda_min); % lambda should not become too close
   % to zero to prevent it from eventually becoming exactly zero - because 
   % zero is the value from which it never can escape anymore via
   % multiplication by 4
 elseif( Delta < 0.25 )
  lambda = 4.0 * lambda; 
  lambda = min(lambda, lambda_max); % lambda should not exceed some 
                                    % ceiling value  
 end
 
else % Delta<0 -> the error would have been increased with this step, so we
     %            keep the old parameters and increase lambda
 gradient  = calculateGradient(parameters, X, w, minBin, maxBin);  
 direction = -gradient;
 beta      = 0;
 lambda    = 4.0 * lambda;
 lambda    = min(lambda, lambda_max); % lambda should not exceed some 
                                      % ceiling value
 newParameters = parameters;
 disp('error would have been increased -> no update step has been made');
 disp(' lambda was increased, algorithm was reset'); 
 return;
 
end

% evaluate the new gradient-vector (remembering the old one for the
% calculation of beta):
gradient_old = gradient;
gradient     = calculateGradient(newParameters, X, w, minBin, maxBin);

% calculate the weight for the old search direction (beta) with one of the 
% 4 possible formulas:
switch( betaFormula )
 case 1 % Polak-Ribiere form
  beta = gradient'*(gradient-gradient_old) / (gradient_old'*gradient_old);
 case 2 % Hestenes-Stiefel form
  beta = gradient'*(gradient-gradient_old) / ...
         (direction_old'*(gradient-gradient_old));
 case 3 % Fletcher-Reeves form
  beta = gradient'*gradient / (gradient_old'*gradient_old);
 case 4
  beta = gradient'*H_times_d_old / (direction_old'*H_times_d_old);
end
% ensure, that beta is always positive:
if(beta<0)
 beta = 0;
end

% experimental: scale beta according to how close Delta is to one:
% beta = beta * (1-abs(1-Delta));

% reset the algorithm every D iterations with D being the dimensionality of
% the parameter-space:
if( mod(iteration,length(parameters))==0 && withReset==true )
 beta = 0;
end

% calculate the new search direction for the next iteration (remembering
% the old  one for the calculation of beta via Hestenes-Stiefel):
direction_old = direction;
direction     = -gradient + beta*direction_old;

% print some info from time to time:
if( mod(iteration, 25) == 0 )
 disp('------------------------------------');
 disp('Scaled Conjugate Gradient Algorithm:'); 
 disp(strcat('iteration = ', num2str(iteration)));
 disp(strcat('error     = ', num2str(error)));
 disp(strcat('alpha     = ', num2str(alpha)));
 disp(strcat('beta      = ', num2str(beta)));
 disp(strcat('delta     = ', num2str(delta))); 
 disp(strcat('Delta     = ', num2str(Delta))); 
 disp(strcat('lambda    = ', num2str(lambda)));  
 disp('------------------------------------'); 
end;
 
 
 
 
 
 
 
 
 
 
 