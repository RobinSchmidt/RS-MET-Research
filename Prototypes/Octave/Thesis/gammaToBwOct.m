function bwOct = gammaToBwOct(gamma, OmegaC);

% This function converts the "gamma"-parameter together with a normalized
% radian center frequency of a parametric equalizer (see Robert Bristow 
% Johnson's paper "The Equivalence of Various Methods of Computing Biquad
% Coefficients for Audio Parametric Equalizers" for details) to a bandwidth
% given in octaves.
%
% usage: 
%  bwOct = bwOctToGamma(gamma, OmegaC);
%
% input-variables:
%  -gamma: the gamma-parameter
%  -OmegaC: the normalized radien center frequency of the equalizer.
%
% output-variables:
%  -bwOct: bandwidth given in octaves

%--------------------------------------------------------------------------

bwOct =  (2*sin(OmegaC)./(log(2)*OmegaC)).*asinh(gamma./sin(OmegaC));