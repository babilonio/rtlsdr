#!/usr/bin/env python
from rtlsdr import RtlSdrTcpClient

client = RtlSdrTcpClient(hostname='192.168.0.155', port=12345)
client.center_freq = 100e6
data = client.read_samples()
print data
