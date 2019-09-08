clear all;

% define the desired parameters of the filter:
sampleRate = 44100;          % sample-rate in Hz
bw_min     = 1/8;
bw_max     = 1;
bw_step    = bw_min;

g_dB       = 20;
g          = 10^(g_dB/20);
centerFreq = 1*sampleRate/4; % center frequency of the equalizer
OmegaC     = 2*pi*centerFreq/sampleRate;
bw         = bw_min;            % bandwidth in octaves
gamma      = sinh(0.5*log(2)*bw*OmegaC/sin(OmegaC))*sin(OmegaC);

% calculate the number of EQ-curves to draw:
numCurves  = round( (bw_max-bw_min)/bw_step ) + 1;

% allocate vectors to hold the curves:
stepsize   = pi/1024;  % the frequency-axis sampling interval
Omegas     = 0:stepsize:(pi-stepsize);
eqCurves   = zeros(length(Omegas),numCurves);

% generate the curves:
bw = bw_min;
for k=1:numCurves
 gamma         = sinh(0.5*log(2)*bw*OmegaC/sin(OmegaC))*sin(OmegaC);
 eqCurves(:,k) = generateEqualizerCurve(1024, [1; g; OmegaC; gamma]);
 bw            = bw + bw_step;
end

% convert to decibels:
eqCurves_dB = 20*log10(eqCurves);

% plot the curves:
figure;
plot(Omegas, eqCurves_dB, 'k');
 xlabel('\Omega (normierte Kreisfrequenz)');
 ylabel('Amplitude in dB');
 axis([0 pi -1 21]);
 grid on;
