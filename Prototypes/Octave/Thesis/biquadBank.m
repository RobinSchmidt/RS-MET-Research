function Y = biquadBank(B0s, B1s, B2s, A0s, A1s, A2s, Gs, X)

% This function implements a parallel connection of biquad filters, each 
% with its own gain-factor g. The difference equation of one such a 
% filter is:
%
% y[n] = g * ( b0*x[n] + b1*x[n-1] + b1*x[n-2]
%                      - a1*y[n-1] - a2*y[n-1] )
%
% usage:
%  Y = biquadBank(B0s, B1s, B2s, A0s, A1s, A2s, Gs, X)
%
% input-variables:
%  -B0s: vector of the b0-coefficients for the individual stages
%  -B1s: vector of the b1-coefficients for the individual stages
%  -B2s: vector of the b2-coefficients for the individual stages
%  -A0s: vector of the a0-coefficients for the individual stages
%  -A1s: vector of the a1-coefficients for the individual stages
%  -A2s: vector of the a2-coefficients for the individual stages
%  -Gs:  vector with a gain factor for the individual stages
%  -X:   the input signal
%
% output-variables:
%  -Y: the output signal

%--------------------------------------------------------------------------

Y = zeros(length(X),1);

S = length(B0s);     % number of biquad-stages
for s=1:S
 B = [B0s(s); B1s(s); B2s(s)];
 A = [A0s(s); A1s(s); A2s(s)];
 G = Gs(s);
 Y = Y + G * filter(B,A,X);  
end