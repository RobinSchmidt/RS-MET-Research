clear all;

% define the desired parameters of the filter:
sampleRate = 44100;          % sample-rate in Hz

centerFreq_min  = 0.5*sampleRate/8;              % minimum center-freq of the curve-family
centerFreq_max  = 0.5*sampleRate-centerFreq_min; % maximum center-freq of the curve-family
centerFreq_step = centerFreq_min;                % stepsize between the curves

g_dB       = 20;
g          = 10^(g_dB/20);
centerFreq = centerFreq_min; 
OmegaC     = 2*pi*centerFreq/sampleRate;
bw         = 1/2;            % bandwidth in octaves
gamma      = sinh(0.5*log(2)*bw*OmegaC/sin(OmegaC))*sin(OmegaC);

% calculate the number of EQ-curves to draw:
numCurves  = round( (centerFreq_max-centerFreq_min)/centerFreq_step ) + 1;

% allocate vectors to hold the curves:
stepsize   = pi/1024;  % the frequency-axis sampling interval
Omegas     = 0:stepsize:(pi-stepsize);
eqCurves   = zeros(length(Omegas),numCurves);

% generate the curves:
centerFreq = centerFreq_min; 
for k=1:numCurves
 OmegaC        = 2*pi*centerFreq/sampleRate;
 eqCurves(:,k) = generateEqualizerCurve(1024, [1; g; OmegaC; gamma]);
 centerFreq    = centerFreq + centerFreq_step;
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
 
