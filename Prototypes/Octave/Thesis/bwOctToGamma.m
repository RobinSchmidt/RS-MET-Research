function gamma = bwOctToGamma(bwOct, OmegaC);

% This function converts a bandwidth given in octaves together with
% a normalized radian center frequency to the "gamma"-parameter in a
% parametric equalizer (see Robert Bristow Johnson's paper "The Equivalence
% of Various Methods of Computing Biquad Coefficients for Audio Parametric
% Equalizers" for details)
%
% usage: 
%  gamma = bwOctToGamma(bwOct, OmegaC);
%
% input-variables:
%  -bwOct: bandwidth given in octaves
%  -OmegaC: the normalized radien center frequency of the equalizer.
%
% output-variables:
%  -gamma: the gamma-parameter

%--------------------------------------------------------------------------

gamma = sinh(0.5*log(2)*bwOct.*OmegaC./sin(OmegaC)).*sin(OmegaC);