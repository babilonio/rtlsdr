#!/usr/bin/python
from cmath import exp, phase
from math import sin, cos, pi, atan2

def cumsum(x, i):
	return sum(x[0:i+1]);	

def conj( c):
	con = c.real - 1j*c.imag;
	return con
                       
Fs = 100.0;                                      
Ts = 1/Fs;                                     
fc = 20;    
fd = 25.0;
kf = fd/Fs;

# error en el oscilador del demodulador 
demod_fc_error = 0.0;

def ind(t):
	return int(round(t*Fs))
           
                                 
time = range(0, int( (0.5-Ts)*Fs) ,int(round(Ts*Fs) ));     
time = [t / Fs for t in time]                   

TestFreq = 4.0;   
msg = [ (sin(2*pi*TestFreq*t) + sin(2*pi*2*TestFreq*t)) for t in time ];
max_msg = max(msg)
norm_msg = [m / max_msg for m in msg];


fmmsg = [0] * len(time)
csum = [0] * len(time)
y = [0] * len(time)
w = [0] * len(time)
v = [0] * len(time)
error = [0] * len(time)

for t in time:
	csum[ind(t)] = cumsum(norm_msg, ind(t))

for t in time:
	fmmsg[ind(t)] = exp( 1j*( 2*pi*fc*t + 2*pi*kf*csum[ind(t)] ));

for t in time:
	y[ind(t)] = fmmsg[ind(t)] * exp( -1j*2*pi*fc*(1 - demod_fc_error)*t);


itertime = iter(time)
next(itertime)
for t in itertime:
	w[ind(t)] = y[ind(t)] * conj( y[ind(t) - 1] );

for t in time:
	v[ind(t)] = atan2(w[ind(t)].imag, w[ind(t)].real)

max_v = max(v);
norm_v = [m / max_v for m in v];

for t in time:
	error[ind(t)] = abs(norm_msg[ind(t)] - norm_v[ind(t)])

print max(error)
