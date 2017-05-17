#COMMON IMPORTS
from __future__ import print_function
import sys
import itertools
import time
import argparse

#SCIENCE IMPORTS
import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin, resample, decimate
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt

#RTLSDR IMPORTS
from pylab import *
from rtlsdr import *

#AUX
subplot_counter = 210

#FUNCTIONS
def conj( c):
	con = c.real - 1j*c.imag;
	return con

def plot_psd( signal, f):
    global subplot_counter
    plt.figure(1)
    subplot_counter +=1
    plt.subplot(subplot_counter)

    # use matplotlib to estimate and plot the PSD
    psd(signal, NFFT=1024, Fs=f)
    xlabel('Frequency (MHz)')
    ylabel('Relative power (dB)')
    return

def plot_constelation( signal):
    global subplot_counter
    plt.figure(1)
    subplot_counter +=1
    plt.subplot(subplot_counter)
    plt.scatter(signal.real, signal.imag, alpha=0.05)
    return

Fs = int(1102500) # This is for getting 44.1KHz as audio rate later
N = int(8192000)  # must be integer, would need to be checked for remainder (N += 1 if necessary)
F_offset = 0 #250000         # Offset to capture at
F_bw = 200e3 # FM Signal bandwidth

# PARSE INPUT
parser = argparse.ArgumentParser(description='Demodulate WBFM-stereo signal')
parser.add_argument("freq", type=float, help='broadcast signal frequency')
args = parser.parse_args()

carrier_freq = args.freq

# DEVICE SETTINGS
sdr = RtlSdr()
sdr.sample_rate = Fs # Hz
sdr.center_freq = carrier_freq - F_offset # Hz
# sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'
print("Reading ", N, " samples, ", N/(1.0*Fs), " seconds")
start = time.time()
samples = sdr.read_samples(N)
end = time.time()
print("Reading time: ", end - start)



# DE_OFFSET
# To mix the data down, generate a digital complex exponential
# (with the same length as x1) with phase -F_offset/Fs
carrier_freq1 = np.exp(-1.0j*2.0*np.pi* F_offset/Fs*np.arange(len(samples)))
# Now, just multiply x1 and the digital complex expontential
samples2 = samples * carrier_freq1
plot_psd(samples2, Fs)
#DOWNSAMPLE AND CHANNEL SELECTION FILTER
start = time.time()
b = firwin(10,  F_bw/Fs)
downsample_rate = int(Fs / F_bw)
Fs_downed = (Fs / downsample_rate);
filtered = np_convolve(b, samples2)
decimated = filtered[0::downsample_rate]
end = time.time()
print("DOWNSAMPLE AND CHANNEL SELECTION FILTER time: ", end - start)


#FM DEMODULATION
start = time.time()
v = [0] * len(decimated)
prev_s = 0
for i, s in enumerate(decimated):
    w = s * conj(prev_s)
    v[i] = np.arctan2(w.imag, w.real)
    prev_s = s
end = time.time()
print("DEMODULATION time: ", end - start)
plot_psd(v, Fs_downed)



# MONO CHANNEL
monof = 15e3 / (Fs_downed/2);
bm = firwin(64,  monof);
mono = np_convolve(bm, v)

# DEEMPH FILTER
tau = 75e-6 * Fs_downed;
decay = exp(-1/tau)
# y[n] = (1 - d)*x[n] + d*y[n-1]
b = [1 - decay]
a = [1, -decay]
mono_deemphasized = lfilter(b,a,mono)

# PILOT
pilotf = 19e3 / (Fs_downed/2);
pilotf_sub = 18.9e3 / (Fs_downed/2);
pilotf_sup = 19.1e3 / (Fs_downed/2);

bp = firwin(64,  [pilotf_sub, pilotf_sup], pass_zero=False);
pilot = np_convolve(bp, v)

doubled_pilot = np.concatenate( (pilot[0::2], pilot[0::2]) )

# STEREO CHANNEL
start = time.time()
stereof_sub = 23e3 / (Fs_downed/2);
stereof_sup = 53e3 / (Fs_downed/2);
bs = firwin(64,  [stereof_sub, stereof_sup], pass_zero=False);

stereo = np_convolve(bs, v)
stereo_base = stereo * doubled_pilot[:len(stereo)];
stereo_deemphasized = lfilter(b,a,stereo_base)

#AUDIO CHANNELS
leftchan = mono_deemphasized + stereo_deemphasized[:len(mono_deemphasized)]
rightchan = mono_deemphasized - stereo_deemphasized[:len(mono_deemphasized)]

# Find a decimation rate to achieve audio sampling rate between 44-48 kHz
audio_freq = 44100
dec_audio = int(Fs_downed/audio_freq)
Fs_audio = Fs_downed / dec_audio

print("AUDIO SAMPLE RATE: ", Fs_audio)

audio_left = decimate(leftchan, dec_audio)
audio_right = decimate(rightchan, dec_audio)

# Scale audio to adjust volume
audio_left *= 10000 / np.max(np.abs(audio_left))
audio_right *= 10000 / np.max(np.abs(audio_right))

audio_stereo = np.array(list(itertools.chain.from_iterable(zip(audio_left,audio_right))))
# Save to file as 16-bit signed single-channel audio samples
#x7.astype("int16").tofile(sys.stdout)
audio_stereo.astype("int16").tofile("wbfm-stereo.raw")


show();
