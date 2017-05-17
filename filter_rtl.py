from __future__ import print_function
from pylab import *
from rtlsdr import *
import sys
import itertools

import time

import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin, resample, decimate
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt


def conj( c):
	con = c.real - 1j*c.imag;
	return con

sdr = RtlSdr()

Fs = int(1140000) # This is for getting 44KHz as audio rate later
fc = 99.8e6
F_offset = 0#250000         # Offset to capture at

# N = int(8192)


f_cut = 200e3
b = firwin(10,  f_cut/Fs)
downsample_rate = int(Fs / f_cut)


N = int(8192000)  # must be integer, would need to be checked for remainder (N += 1 if necessary)

start = time.time()
# configure device
sdr.sample_rate = Fs # Hz
sdr.center_freq = fc - F_offset # Hz
# sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'
print("Reading ", N, " samples, ", N/(1.0*Fs), " seconds")
samples = sdr.read_samples(N)
# read_size = 8192
# num_read = 0 # number of samples read
# samples = None
# while num_read <= N:
#     sweep = sdr.read_samples(read_size)
#     if samples is None:
#         samples = sweep
#     else:
#         samples = np.concatenate((samples, sweep))
#     num_read += read_size
# if samples.size > N:
#     samples = samples[:N]  # I think that slice expression is correct, would have to verify
end = time.time()
print("Reading time: ", end - start)

start = time.time()
# To mix the data down, generate a digital complex exponential
# (with the same length as x1) with phase -F_offset/Fs
fc1 = np.exp(-1.0j*2.0*np.pi* F_offset/Fs*np.arange(len(samples)))
# Now, just multiply x1 and the digital complex expontential
samples2 = samples * fc1
print("Downconversion time: ", end - start)

start = time.time()
filtered = np_convolve(b, samples2)
end = time.time()
print("Filter time: ", end - start)


start = time.time()
decimated = filtered[0::downsample_rate]
end = time.time()
print("Decimate time: ", end - start)


start = time.time()
v = [0] * len(decimated)
prev_s = 0

for i, s in enumerate(decimated):
    w = s * conj(prev_s)
    v[i] = np.arctan2(w.imag, w.real)
    prev_s = s
end = time.time()
print("discriminator time: ", end - start)

# dec_factor = sdr.sample_rate / 48e3
# dec = decimate(v, 10)
# r = resample(dec, int(len(dec)/48e3) )

plt.figure(1)
plt.subplot(211)
#
# # use matplotlib to estimate and plot the PSD
psd(v, NFFT=1024, Fs=f_cut)
# xlabel('Frequency (MHz)')
# ylabel('Relative power (dB)')
#
# # show()
#
# # plt.subplot(212)
# # plt.scatter(decimated.real, decimated.imag, alpha=0.05)
#
# show()

nFs = (Fs / downsample_rate);

# MONO
monof = 15e3 / (nFs/2);
bm = firwin(64,  monof);
mono = np_convolve(bm, v)

# DEEMPH
start = time.time()

#tau
tau = 75e-6 * nFs;
#decay
d = exp(-1/tau)
# y[n] = (1 - d)*x[n] + d*y[n-1]
b = [1 - d]
a = [1, -d]
deemphasized = lfilter(b,a,mono)
end = time.time()
print("deemphasis time: ", end - start)

# PILOT
start = time.time()
pilotf = 19e3 / (nFs/2);
pilotf_sub = 18.9e3 / (nFs/2);
pilotf_sup = 19.1e3 / (nFs/2);

bp = firwin(64,  [pilotf_sub, pilotf_sup], pass_zero=False);
pilot = np_convolve(bp, v)

doubled_pilot = np.concatenate( (pilot[0::2], pilot[0::2]) )
end = time.time()
print("pilot filter time: ", end - start)

# STEREO
start = time.time()
stereof_sub = 23e3 / (nFs/2);
stereof_sup = 53e3 / (nFs/2);
bs = firwin(64,  [stereof_sub, stereof_sup], pass_zero=False);

stereo = np_convolve(bs, v)

stereo_base = stereo * doubled_pilot[:len(stereo)];
stereo_deemphasized = lfilter(b,a,stereo_base)

leftchan = deemphasized + stereo_deemphasized[:len(deemphasized)]
rightchan = deemphasized - stereo_deemphasized[:len(deemphasized)]
end = time.time()
print("stereo filter time: ", end - start)

plt.figure(1)
plt.subplot(212)
# use matplotlib to estimate and plot the PSD
psd(leftchan, NFFT=1024, Fs=f_cut)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')

show();

# Find a decimation rate to achieve audio sampling rate between 44-48 kHz
audio_freq = 44100
dec_audio = int(f_cut/audio_freq)
Fs_audio = f_cut / dec_audio

print(Fs_audio)

x_left = decimate(leftchan, dec_audio)
x_right = decimate(rightchan, dec_audio)

# Scale audio to adjust volume
x_left *= 10000 / np.max(np.abs(x_left))
x_right *= 10000 / np.max(np.abs(x_right))

x7 = np.array(list(itertools.chain.from_iterable(zip(x_left,x_right))))
# Save to file as 16-bit signed single-channel audio samples
#x7.astype("int16").tofile(sys.stdout)
x7.astype("int16").tofile("wbfm-stereo.raw")
