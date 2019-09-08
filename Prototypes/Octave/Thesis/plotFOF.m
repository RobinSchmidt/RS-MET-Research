clear all;

% This script plots a formant-wave-function (Forme de'Onde Formantique,
% FOF) of the form x[n] = A * exp(-alpha*n) * sin(omega*n + phi).

% assign the parameters:
N     = 1024;    % the number of bins
A     = 1;       
alpha = 0.005;
omega = pi/50;
phi   = pi/2;

impRespViaFormula = zeros(N,1);
for n=1:N
 impRespViaFormula(n) = A * exp(-alpha*(n-1)) * sin(omega*(n-1)+phi); 
end

figure;
 plot(0:N-1, impRespViaFormula);
 grid on;
 axis([0 N-1 -1.1 1.1]);
 xlabel('n (Sample-Nummer)');
 ylabel('h[n] (Impulsantwort)'); 











 
