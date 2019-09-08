clear all;

% assign the sample-rate and related stuff:
f_s          = 44100;
numBins      = 1024;
binFreqs     = ((0:numBins-1)/numBins) * (f_s/2);

% assign the gain-factor and the range and stepsize for the zero (and
% calculate the number of filter-curves from that):
g          = 1;
minZero    = -1.0;
maxZero    = +1.0;
stepSize   =  0.25; 
numFilters = (maxZero-minZero)/stepSize+1;

% calculate the normalized radian frequencies:
Omegas = 0:(1/(numBins-1)):1;
Omegas = pi*Omegas;

% allocate a matrix to hold the curves:
magnitudes = zeros(length(Omegas),numFilters);

% generate the curves:
zero = minZero;
for i=1:numFilters

 % calculate the position of the current zero:
 zero = minZero + (i-1)*stepSize
 
 % calculate the filter coefficients:
 b0 = 1/g;
 b1 = -zero/g;
 b2 = 0;

 % evaluate the magnitude response:
 for k=1:length(Omegas)
  Omega  = Omegas(k);
  num    = g * (b0^2+b1^2+b2^2+2*cos(Omega)*(b0*b1+b1*b2)+2*cos(2*Omega)*b0*b2); % numerator
  den    = 1;           % denominator
  magnitudes(k,i) = sqrt(num/den);
 end
 
end

% convert to decibels:
dBmagnitudes = 20*log10(magnitudes);
 
% plot the curves:
plot(Omegas, dBmagnitudes, 'k');
 xlabel('\Omega (normierte Kreisfrequenz)');
 ylabel('Amplitude in dB');
 axis([0 pi -30 10]);
 grid on;
 