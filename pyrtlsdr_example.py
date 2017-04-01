from __future__ import print_function
from rtlsdr import RtlSdr

sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.048e6  # Hz
sdr.center_freq = 100.4e6     # Hz
sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'

samples = sdr.read_samples(256*1024)

for s in samples:
	print(s)
#f = open('datasdr', 'w')
#f.write(samples)
