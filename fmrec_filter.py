#!/usr/bin/python
from cmath import exp, phase
from math import sin, cos, pi, atan2
import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin

def conj( c):
	con = c.real - 1j*c.imag;
	return con
                       
Fs = 100.0;                                      
Ts = 1/Fs;                                     
fc = 20;    
fd = 25.0;
kf = fd/Fs;

#Create a FIR filter
b = firwin(2, 0.99, width=0.05, pass_zero=True)
# error en el oscilador del demodulador 
demod_fc_error = 0.0;

def ind(t):
	return int(round(t*Fs))
           
                                 
time = range(0, int( (0.5-Ts)*Fs) ,int(round(Ts*Fs) ));     
          
TestFreq = 4.0;   
prev_csum = 0
prev_y =0+0j
msg = [0] * len(time)
v = [0] * len(time)
fmmsg = np.zeros(len(time), dtype=complex)

for n in time:
	t = n / Fs
	msg[n] = sin(2*pi*TestFreq*t) + sin(2*pi*2*TestFreq*t)
	csum = msg[n] + prev_csum
	prev_csum = csum
	fmmsg[n] = exp( 1j*( 2*pi*fc*t + 2*pi*kf*csum ));

npconv_result = fmmsg # np_convolve(fmmsg, b, mode='valid') 

for n in range(0, len(npconv_result)):
	y = npconv_result[n] * exp( -1j*2*pi*fc*(1 - demod_fc_error)*t);
	w = y * conj(prev_y );
	prev_y = y
	v[n] = atan2(w.imag, w.real)


print msg[0:10]
print v[0:10]
