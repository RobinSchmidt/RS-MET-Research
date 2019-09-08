function [A, alpha, omega, phi] = filterCoeffsToFofParams(g, b1, a1, a2)

% This function converts the set of coefficients (g, b1, a1, a2) of a
% two-pole-one-zero filter realizing the difference equation:
%
% y[n] = g * ( x[n] + b1*x[n-1] - a1*y[n-1] - a2*y[n-2] )
%
% to a set of FOF-parameters (A, alpha, omega, phi) representing the damped
% sinosoid:
%
% h[n] = A * exp(-alpha*n) * sin(omega*n + phi)
%
% which is the impulse-response of the filter.
%
% usage: 
%  [A, alpha, omega, phi] = filterCoeffsToFofParams(g, b1, a1, a2)
%
% input-variables:
%  -g:  the overall gain of the filter
%  -b0: the b0-coefficient (determining the position of the zero 
%       as z0 = -b0)
%  -a1: the a1-coefficient
%  -a2: the a2-coefficient
%
% output-variables:
%  -A:     the amplitude of the FOF
%  -alpha: the damping constant
%  -omega: the normalized radian frequency
%  -phi:   the start-phase of the sinosoid

%--------------------------------------------------------------------------

P      = sqrt(a2);
alpha  = -log(P);
omega  = acos(-a1./(2.*P))
r_im   = - (sin(2*omega)+2.*(b1./P).*sin(omega))./ (2*(1-cos(2*omega)));
R      = sqrt(0.25 + r_im.^2);
varphi = atan2(r_im,0.5);
phi    = varphi + pi./2;
A      = 2*g.*R;