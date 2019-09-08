clear all;

% assign sample-rate and resonant frequency:
sampleRate = 44100;          % sample-rate in Hz
centerFreq = 1*sampleRate/8; % center frequency of the pole-pair in Hz

% assign the range and stepsize for the pole-radius (and calculate the
% number of filter-curves from that):
minRadius      = 0.29;       % lowest radius of the pole-pair
maxRadius      = 0.99;
radiusStepSize = 0.1;
numFilters     = round((maxRadius-minRadius)/radiusStepSize)+1;

% calculate the normalized radian frequencies:
stepsize   = pi/1024;
Omegas     = 0:stepsize:(pi-stepsize);

% allocate a matrix to hold the curves:
magnitudes = zeros(length(Omegas),numFilters);

% generate the curves:
radius = minRadius;
for i=1:numFilters
 
 % convert the center-frequency into a normalized center-frequency in
 % radians/sec (a frequency between 0 and pi):
 Omega0 = (2*pi*centerFreq)/sampleRate;
 
 % calculate the biquad-filter coefficients:
 b0 = 1;
 b1 = 0;
 b2 = 0;
 a0 = 1;
 a1 = -2*radius*cos(Omega0);
 a2 = radius^2;
 
 % evaluate the magnitude response: 
 for k=1:length(Omegas)
  Omega  = Omegas(k);
  num    = b0^2+b1^2+b2^2+2*cos(Omega)*(b0*b1+b1*b2)+2*cos(2*Omega)*b0*b2; % numerator
  den    = a0^2+a1^2+a2^2+2*cos(Omega)*(a0*a1+a1*a2)+2*cos(2*Omega)*a0*a2; % denominator
  magnitudes(k,i) = sqrt(num/den);
 end
 
 % increment the radius:
 radius = radius + radiusStepSize;
 
end

% convert to decibels:
dBmagnitudes = 20*log10(magnitudes);
 
% plot the curves:
plot(Omegas, dBmagnitudes, 'k');
 xlabel('\Omega (normierte Kreisfrequenz)');
 ylabel('Amplitude in dB');
 axis([0 pi -12 42]);
 grid on;




