clear all;

% assign the sample-rate and related stuff:
f_s          = 44100;
numBins      = 1024;
binFreqs     = ((0:numBins-1)/numBins) * (f_s/2);

% assign the pole-radius and angle:
poleRadius = 0.99;
poleAngle  = pi/2;
poles      = poleRadius.*exp(j*poleAngle);
poles      = [poles; conj(poles)];

% assign the gain-factor and the range and stepsize for the zero (and
% calculate the number of filter-curves from that):
g          = 1;
%zero       = +1.0;
minZero    = -2.0;
maxZero    = +2.0;
stepSize   =  0.25; 
numFilters = (maxZero-minZero)/stepSize+1;

% calculate the normalized radian frequencies:
stepsize   = pi/1024;
Omegas     = 0:stepsize:(pi-stepsize);

% allocate a matrix to hold the curves:
magnitudes = zeros(length(Omegas),numFilters);

% generate the curves:
zero = minZero;
for i=1:numFilters

 % calculate the position of the current zero: 
 zero = minZero + (i-1)*stepSize
 
 % calculate the biquad-filter coefficients:
 b0 = 1/g;
 b1 = -zero/g;
 b2 = 0;
 a0 = 1;
 a1 = -2*poleRadius*cos(poleAngle);
 a2 = poleRadius^2;
 
 % evaluate the phase-response: 
 for k=1:length(Omegas)
  Omega  = Omegas(k);
  z      = exp(j*Omega);
  num    = g * (b0 + b1*z^(-1) + b2*z^(-2)); % numerator
  den    =      a0 + a1*z^(-1) + a2*z^(-2);  % denominator   
  H      = num/den;
  %magnitudes(k,i) = abs(H);
  phases(k,i)     = angle(H);  
 end
 
end

% convert to decibels:
dBmagnitudes = 20*log10(magnitudes);
 
% plot the curves:
plot(Omegas, phases, 'k');
 xlabel('\Omega (normierte Kreisfrequenz)');
 ylabel('Phase');
 axis([0 pi -3.5 3.5]);
 grid on;