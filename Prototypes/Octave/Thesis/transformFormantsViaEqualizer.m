function outputSignal = transformFormantsViaEqualizer(inputSignal, oldEqualizerParameters, newEqualizerParameters)

% This function transforms the formants of a signal by means of two
% parametric equalizer-chains. The first one is used for whitening the
% signal by applying the inverse equalizer (assuming that the first vector
% of equalizer-parameters was obtained by fitting the equalizers magnitude
% response to the spectral envelope of the signal). The second one then is
% used to apply the new formants to the signal.
%
% usage: 
%  outputSignal = transformFormantsViaEqualizer(inputSignal, ...
%                  oldEqualizerParameters, newEqualizerParameters)
%
% input-variables:
%  -inputSignal: the input signal
%  -oldEqualizerParameters: vector of the parameters of the equalizer which
%    should be undone - that is: the inverse of this equalizer will be
%    applied as whitening-filter. These parameters usually come from
%    fitting an equalizers magnitude-response to the spectral envelope of
%    the signal.
%  -newEqualizerParameters: These are the parameters of the equalizer which
%    will be applied after the whitening via the inverse old equalizer has
%    been done.
%
% output-variables:
%  -outputSignal: the output-signal

%--------------------------------------------------------------------------

% extract the number of filter-stages for the old equalizer (which should
% be undone for whitening)
oldNumStages = round((length(oldEqualizerParameters)-1)/3);

% decompose the old parameter-vector:
G_old      = oldEqualizerParameters(1);
g_old      = oldEqualizerParameters(2:(oldNumStages+1));
OmegaC_old = oldEqualizerParameters((oldNumStages+2):(2*oldNumStages+1));
gamma_old  = oldEqualizerParameters((2*oldNumStages+2):(3*oldNumStages+1));

% extract the number of filter-stages for the new equalizer (which should
% be applied to the whitened signal):
newNumStages = round((length(newEqualizerParameters)-1)/3);

% decompose the new parameter-vector:
G_new      = newEqualizerParameters(1);
g_new      = newEqualizerParameters(2:(newNumStages+1));
OmegaC_new = newEqualizerParameters((newNumStages+2):(2*newNumStages+1));
gamma_new  = newEqualizerParameters((2*newNumStages+2):(3*newNumStages+1));

% whiten the input (undo the effect of the old (estimated) eq):
whiteSignal = equalizeSignal(inputSignal, [1; 1./g_old; OmegaC_old; gamma_old]);

% apply the new formants to the whitened signal:
outputSignal = equalizeSignal(whiteSignal, [1; g_new; OmegaC_new; gamma_new]);
