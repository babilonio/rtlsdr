#!python
#%matplotlib inline
from __future__ import print_function
import time

import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d
#import matplotlib.pyplot as plt

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
    print ("Create a FIR filter.")
    b = firwin(ntaps, [0.05, 0.95], width=0.05, pass_zero=False)

    print (" --- signal.convolve ---")
    tstart = time.time()
    conv_result = sig_convolve(x, b[np.newaxis, :], mode='valid')
    conv_time.append(time.time() - tstart)

    print ("--- numpy.convolve ---")
    tstart = time.time()
    npconv_result = np.array([np_convolve(xi, b, mode='valid') for xi in x])
    npconv_time.append(time.time() - tstart)

    print ("--- signal.fftconvolve ---")
    tstart = time.time()
    fftconv_result = fftconvolve(x, b[np.newaxis, :], mode='valid')
    fftconv_time.append(time.time() - tstart)

    print  ("--- convolve1d ---")
    tstart = time.time()
    # convolve1d doesn't have a 'valid' mode, so we expliclity slice out
    # the valid part of the result.
    conv1d_result = convolve1d(x, b)[:, (len(b)-1)//2 : -(len(b)//2)]
    conv1d_time.append(time.time() - tstart)

    print( "--- lfilter ---")
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


print (conv_time )
print( npconv_time )
print (fftconv_time )
print (conv1d_time )
print (lfilt_time )
