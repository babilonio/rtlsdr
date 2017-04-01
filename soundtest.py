from math import cos, pi
import time
import struct

Fs = 48000.0
freq = 440.0
t = 0
inc = 1/Fs

while True:
    s = cos( 2*pi*freq*t)
        #byt = s.to_bytes(2, byteorder='little')
    sample = int(s * 0x7f + 0x80)
    byt = sample.to_bytes(2, byteorder='little')
    for b in byt:
        print(chr(b), end='')
    t += inc
    # time.sleep(inc)
