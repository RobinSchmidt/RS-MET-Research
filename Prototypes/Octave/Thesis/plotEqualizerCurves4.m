clear all;

% define the desired parameters of the filter:
sampleRate = 44100;          % sample-rate in Hz

g_dB       = 20;
g          = 10^(g_dB/20);
centerFreq = 31.25; 
OmegaC     = 2*pi*centerFreq/sampleRate;
bw         = 1/1;            % bandwidth in octaves
gamma      = sinh(0.5*log(2)*bw*OmegaC/sin(OmegaC))*sin(OmegaC);
numCurves  = 10;

% allocate vectors to hold the curves:
stepsize   = pi/65536;  % the frequency-axis sampling interval
Omegas     = 0:stepsize:(pi-stepsize);
binFreqs   = sampleRate*Omegas / (2*pi);
eqCurves   = zeros(length(Omegas),numCurves);

% generate the curves:
for k=1:numCurves
 OmegaC        = 2*pi*centerFreq/sampleRate;
 gamma         = sinh(0.5*log(2)*bw*OmegaC/sin(OmegaC))*sin(OmegaC); 
 eqCurves(:,k) = generateEqualizerCurve(65536, [1; g; OmegaC; gamma]);
 centerFreq    = 2*centerFreq;
end

% convert to decibels:
eqCurves_dB = 20*log10(eqCurves);

% plot the curves:
figure;
semilogx(binFreqs, eqCurves_dB, 'k');
 xlabel('f in Hz (f_s=44100)');
 ylabel('Amplitude in dB');
 axis([7.8125 22050 -1 21]);
 grid on; 
