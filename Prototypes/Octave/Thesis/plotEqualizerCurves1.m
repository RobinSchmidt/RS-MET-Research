clear all;

% define the desired parameters of the filter:
sampleRate = 44100;          % sample-rate in Hz

g_dB_min   = -20;            % minimum gain of the curve-family
g_dB_max   =  20;            % maximum gain of the curve-family
g_dB_step  =   5;            % stepsize between the curves

g_dB       = g_dB_min;
g          = 10^(g_dB/20);
centerFreq = 1*sampleRate/4; % center frequency of the equalizer
OmegaC     = 2*pi*centerFreq/sampleRate;
bw         = 1/2;            % bandwidth in octaves
gamma      = sinh(0.5*log(2)*bw*OmegaC/sin(OmegaC))*sin(OmegaC);

% calculate the number of EQ-curves to draw:
numCurves  = round( (g_dB_max-g_dB_min)/g_dB_step ) + 1; 

% allocate vectors to hold the curves:
stepsize   = pi/1024;  % the frequency-axis sampling interval
Omegas     = 0:stepsize:(pi-stepsize);
eqCurves   = zeros(length(Omegas),numCurves);

% generate the curves:
g_dB = g_dB_min;
for k=1:numCurves
 g             = 10^(g_dB/20); 
 eqCurves(:,k) = generateEqualizerCurve(1024, [1; g; OmegaC; gamma]);
 g_dB          = g_dB + g_dB_step;
end

% convert to decibels:
eqCurves_dB = 20*log10(eqCurves);

% plot the curves:
figure;
plot(Omegas, eqCurves_dB, 'k');
 xlabel('\Omega (normierte Kreisfrequenz)');
 ylabel('Amplitude in dB');
 axis([0 pi -21 21]);
 grid on;
