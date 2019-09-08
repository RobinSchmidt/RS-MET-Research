function window = cosineSquaredWindow(length);

% This function generates a window-function with the shape of a squared
% cosine-function (in the interval -pi/2 - +pi/2). Such a window has the
% desirable property of adding up to a constant when the hopsize is equal 
% to the blocksize divided by some power of two. The frequency response is
% quite similar to the Hanning-window.
%
% usage:
%  window = cosineSquaredWindow(length);
%
% input-variables:
%  -length: length of the window in samples
%
% input-variables:
%  -window: the window-function itself

%--------------------------------------------------------------------------

window = (0:length-1)';
window = sin(pi * window/length );
window = window.^2;