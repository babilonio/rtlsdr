#!/usr/bin/env python
from rtlsdr import RtlSdrTcpServer

server = RtlSdrTcpServer(hostname='192.168.0.155', port=12345)
server.run_forever()
# Will listen for clients until Ctrl-C is pressed
