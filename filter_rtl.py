from __future__ import print_function
from pylab import *
from rtlsdr import *
import sys

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

Fs = 1e6
fc = 100.4e6
# configure device
sdr.sample_rate = Fs  # Hz
sdr.center_freq = fc     # Hz
# sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'
N = 4*Fs


f_cut = 200e3
b = firwin(10, f_cut/Fs)
downsample_rate = int(Fs / f_cut)




start = time.time()
samples = sdr.read_samples(N)
end = time.time()
print("Reading time: ", end - start)



start = time.time()
filtered = np_convolve(b, samples)
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
#
# dec_factor = sdr.sample_rate / 48e3
# dec = decimate(v, 10)
# r = resample(dec, int(len(dec)/48e3) )

# plt.figure(1)
# plt.subplot(211)
#
# # use matplotlib to estimate and plot the PSD
# psd(v, NFFT=1024, Fs=f_cut)
# xlabel('Frequency (MHz)')
# ylabel('Relative power (dB)')
#
# # show()
#
# plt.subplot(212)
# plt.scatter(decimated.real, decimated.imag, alpha=0.05)
#
# show()

# Find a decimation rate to achieve audio sampling rate between 44-48 kHz
audio_freq = 48000.0
dec_audio = int(f_cut/audio_freq)
Fs_audio = f_cut / dec_audio

x7 = decimate(v, dec_audio)

# Scale audio to adjust volume
x7 *= 10000 / np.max(np.abs(x7))
# Save to file as 16-bit signed single-channel audio samples
x7.astype("int16").tofile(sys.stdout)


print(Fs_audio)
