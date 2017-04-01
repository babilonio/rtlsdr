#!python
#%matplotlib inline
from __future__ import print_function
import time

import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt

# Create the m by n data to be filtered.
m = 1
n = 2 ** 18
x = np.random.random(size=(m, n))

conv_time = []
npconv_time = []
fftconv_time = []
conv1d_time = []
lfilt_time = []

diff_list = []
diff2_list = []
diff3_list = []

ntaps_list = 2 ** np.arange(2, 14)

for ntaps in ntaps_list:
    # Create a FIR filter.
    b = firwin(ntaps, [0.05, 0.95], width=0.05, pass_zero=False)

    # --- signal.convolve ---
    tstart = time.time()
    conv_result = sig_convolve(x, b[np.newaxis, :], mode='valid')
    conv_time.append(time.time() - tstart)

    # --- numpy.convolve ---
    tstart = time.time()
    npconv_result = np.array([np_convolve(xi, b, mode='valid') for xi in x])
    npconv_time.append(time.time() - tstart)

    # --- signal.fftconvolve ---
    tstart = time.time()
    fftconv_result = fftconvolve(x, b[np.newaxis, :], mode='valid')
    fftconv_time.append(time.time() - tstart)

    # --- convolve1d ---
    tstart = time.time()
    # convolve1d doesn't have a 'valid' mode, so we expliclity slice out
    # the valid part of the result.
    conv1d_result = convolve1d(x, b)[:, (len(b)-1)//2 : -(len(b)//2)]
    conv1d_time.append(time.time() - tstart)

    # --- lfilter ---
    tstart = time.time()
    lfilt_result = lfilter(b, [1.0], x)[:, len(b) - 1:]
    lfilt_time.append(time.time() - tstart)

    diff = np.abs(fftconv_result - lfilt_result).max()
    diff_list.append(diff)

    diff2 = np.abs(conv1d_result - lfilt_result).max()
    diff2_list.append(diff2)

    diff3 = np.abs(npconv_result - lfilt_result).max()
    diff3_list.append(diff3)

# Verify that np.convolve and lfilter gave the same results.
print("Did np.convolve and lfilter produce the same results?",)
check = all(diff < 1e-13 for diff in diff3_list)
if check:
    print( "Yes.")
else:
    print( "No!  Something went wrong.")

# Verify that fftconvolve and lfilter gave the same results.
print( "Did fftconvolve and lfilter produce the same results?")
check = all(diff < 1e-13 for diff in diff_list)
if check:
    print( "Yes.")
else:
    print( "No!  Something went wrong.")

# Verify that convolve1d and lfilter gave the same results.
print( "Did convolve1d and lfilter produce the same results?",)
check = all(diff2 < 1e-13 for diff2 in diff2_list)
if check:
    print( "Yes.")
else:
    print( "No!  Something went wrong.")


def timeit(fn, shape, lfilter=False, n_x=2e4, repeats=3):
    x = np.random.rand(int(n_x))
    y = np.random.rand(*shape)
    args = [x, y] if not lfilter else [x, x, y]
    times = []
    for _ in range(int(repeats)):
        start = time.time()
        c = fn(*args)
        times += [time.time() - start]
    return min(times)

npconv_time2, conv_time2, conv1d_time2 = [], [], []
fftconv_time2, sig_conv_time2, lconv_time2 = [], [], []
Ns_1d = 2*np.logspace(0, 4, num=11, dtype=int)
for n in Ns_1d:
    npconv_time2 += [timeit(np_convolve, shape=(n,))]
    conv1d_time2 += [timeit(convolve1d, shape=(n,))]
    fftconv_time2 += [timeit(fftconvolve, shape=(n,))]
    sig_conv_time2 += [timeit(sig_convolve, shape=(n,))]
    lconv_time2 += [timeit(lfilter, shape=(n,), lfilter=True)]

fig = plt.figure(1, figsize=(16, 5.5))
plt.subplot(1, 2, 1)
plt.loglog(ntaps_list, conv1d_time, 'k-p', label='ndimage.convolve1d')
plt.loglog(ntaps_list, lfilt_time, 'c-o', label='signal.lfilter')
plt.loglog(ntaps_list, fftconv_time, 'm-*', markersize=8, label='signal.fftconvolve')
plt.loglog(ntaps_list[:len(conv_time)], conv_time, 'g-d', label='signal.convolve')
plt.loglog(ntaps_list, npconv_time, 'b-s', label='numpy.convolve')
plt.legend(loc='best', numpoints=1)
plt.grid(True)
plt.xlabel('Number of taps')
plt.ylabel('Time to filter (seconds)')
plt.title('Multidimensional timing')

plt.subplot(1, 2, 2)
plt.loglog(Ns_1d, conv1d_time2, 'k-p', label='ndimage.convolve1d')
plt.loglog(Ns_1d, lconv_time2, 'c-o', label='signal.lfilter')
plt.loglog(Ns_1d, fftconv_time2, 'm-*', markersize=8, label='signal.fftconvolve')
plt.loglog(Ns_1d, sig_conv_time2, 'g-d', label='signal.convolve')
plt.loglog(Ns_1d, npconv_time2, 'b-s', label='np.convolve')
plt.grid()
plt.xlabel('Number of taps')
plt.ylabel('Time to filter (seconds)')
plt.title('One dimensional timing')
plt.legend(loc='best')
plt.show()
