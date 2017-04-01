#!/usr/bin/python
from cmath import exp, phase
from math import sin, cos, pi, atan2

def conj( c):
	con = c.real - 1j*c.imag;
	return con
                       
Fs = 2e5;                                      
Ts = 1/Fs;                                     
fc = 20;    
fd = 25.0;
kf = fd/Fs;

# error en el oscilador del demodulador 
demod_fc_error = 0.0;

def ind(t):
	return int(round(t*Fs))
           
                                 
time = range(0, int( (0.5-Ts)*Fs) ,int(round(Ts*Fs) ));     
          
TestFreq = 4.0;   
prev_csum = 0
prev_y =0+0j
v = [0] * len(time)

for n in time:
	t = n / Fs
	msg = sin(2*pi*TestFreq*t) + sin(2*pi*2*TestFreq*t)
	csum = msg + prev_csum
	prev_csum = csum
	fmmsg = exp( 1j*( 2*pi*fc*t + 2*pi*kf*csum ));
	y = fmmsg * exp( -1j*2*pi*fc*(1 - demod_fc_error)*t);
	w = y * conj(prev_y );
	prev_y = y
	v[n] = atan2(w.imag, w.real)

