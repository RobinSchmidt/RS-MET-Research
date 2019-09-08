clear all;

% This script uses MatLab's Symbolic-Math Toolbox to prove, that, for two
% biquad-equalizers designed with Robert Bristow Johnson's equations with
% reciprocal values for the gain-parameters (all other parameters being the
% same), the tansfer-functions are the inverse of each other. Simply launch 
% the script and see that the result of a multiplication of the 
% transfer-functions (called H(z) and I(z)) is unity.

% declare the variables for the equalizer-parameters:
syms gamma g OmegaC;

% assign the biquad-coffiecients to the design-equations
b0 = sym(' (1+gamma*sqrt(g)) / (1+gamma/sqrt(g)) ');
b1 = sym(' (-2*cos(OmegaC))  / (1+gamma/sqrt(g)) ');
b2 = sym(' (1-gamma*sqrt(g)) / (1+gamma/sqrt(g)) ');
a0 = sym('1');
a1 = sym(' (-2*cos(OmegaC))  / (1+gamma/sqrt(g)) ');
a2 = sym(' (1-gamma/sqrt(g)) / (1+gamma/sqrt(g)) ');

% assign the biquad-coffiecients for the equalizer with reciprocal
% gain-values:
b0Inv = sym(' (1+gamma*sqrt(1/g)) / (1+gamma/sqrt(1/g)) ');
b1Inv = sym(' (-2*cos(OmegaC))  / (1+gamma/sqrt(1/g)) ');
b2Inv = sym(' (1-gamma*sqrt(1/g)) / (1+gamma/sqrt(1/g)) ');
a0Inv = sym('1');
a1Inv = sym(' (-2*cos(OmegaC))  / (1+gamma/sqrt(1/g)) ');
a2Inv = sym(' (1-gamma/sqrt(1/g)) / (1+gamma/sqrt(1/g)) ');

% declare the symbolic variable z and establish the z-domain
% transfer-functions of the two equalizer-filters H(z), I(z):
syms z;
H = subs(sym('(b0 + b1*z^(-1) + b2*z^(-2)) / (a0 + a1*z^(-1) + a2*z^(-2))'));
I = subs(sym('(b0Inv + b1Inv*z^(-1) + b2Inv*z^(-2)) / (a0Inv + a1Inv*z^(-1) + a2Inv*z^(-2))'));

% proof that H(z)*I(z) is unity :
H_times_I = subs(sym('H*I'));
H_times_I = simple(H_times_I)
