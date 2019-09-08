function [g, b1, a1, a2] = fofParamsToFilterCoeffs(A, alpha, omega, phi)

% This function converts the set of FOF-parameters (A, alpha, omega, phi)
% representing the damped sinosoid:
%
% h[n] = A * exp(-alpha*n) * sin(omega*n + phi)
%
% to a set of filter coefficients (g, b1, a1, a2) for a two-pole-one-zero
% filter realizing the difference equation:
%
% y[n] = g * ( x[n] + b1*x[n-1] - a1*y[n-1] - a2*y[n-2] )
%
% which has the damped sinoisoid as its impulse response.
%
% usage: 
%  [g, b1, a1, a2] = fofParamsToFilterCoeffs(A, alpha, omega, phi)
%
% input-variables:
%  -A:     the amplitude of the FOF
%  -alpha: the damping constant
%  -omega: the normalized radian frequency
%  -phi:   the start-phase of the sinosoid
%
% output-variables:
%  -g:  the overall gain of the filter
%  -b0: the b0-coefficient (determining the position of the zero 
%       as z0 = -b0)
%  -a1: the a1-coefficient
%  -a2: the a2-coefficient

%--------------------------------------------------------------------------

varphi = phi - pi/2;
P      = exp(-alpha);
r_im   = 0.5 * tan(varphi);
R      = sqrt(0.25 + r_im.^2);
a2     = P.^2;
a1     = -2*P.*cos(omega);
g      = A./(2*R);
b1     = -(P/2).* ( 2*(1-cos(2*omega)).*r_im + sin(2*omega) )./ sin(omega);
