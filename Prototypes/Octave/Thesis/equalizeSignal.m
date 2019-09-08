function outputSignal = equalizeSignal(inputSignal, equalizerParameters)

% This function takes an input-signal and applies N-band parametric
% equalizing to it. The vector of the equalizer-parameters is assumed to be
% in the form: [G; g; OmegaC; bw], where G is an overall gain-factor, g is
% the vector of the N gain-factors at the eq-center frequencies, OmegaC is
% the vector of the N eq-center frequencies (expressed as normalized radian
% frequency) and gamma is the vecctor of the N eq-bandwidth-parameters.
%
% usage: 
%  outputSignal = equalizeSignal(inputSignal, equalizerParameters)
%
% input-variables:
%  -inputSignal: signal to be equalized
%  -parameters: the vector of equalizer-parameters [G; g; OmegaC; gamma]
%
% output-variables:
%  -outputSignal: the equalized signal

%--------------------------------------------------------------------------

% extract the number of filter-stages:
numStages = round((length(equalizerParameters)-1)/3);

% decompose the parameter-vector:
G         = equalizerParameters(1);
g         = equalizerParameters(2:(numStages+1));
OmegaC    = equalizerParameters((numStages+2):(2*numStages+1));
gamma     = equalizerParameters((2*numStages+2):(3*numStages+1));

% calculate the a- and b-coefficients for all the filter stages:
b0 = (1+gamma.*sqrt(g))./(1+gamma./sqrt(g));
b1 = -2*cos(OmegaC)./(1+gamma./sqrt(g)); 
b2 = (1-gamma.*sqrt(g))./(1+gamma./sqrt(g));
a0 = ones(numStages,1);
a1 = b1;
a2 = (1-gamma./sqrt(g))./(1+gamma./sqrt(g));

% apply the filter-stages one after another:
outputSignal = inputSignal;
for p=1:numStages
 b = [b0(p) b1(p) b2(p)];
 a = [a0(p) a1(p) a2(p)];
 outputSignal = filter(b, a, outputSignal); 
end

% apply the overall gain-factor:
outputSignal = G * outputSignal;