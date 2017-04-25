import sys
import array
import time
import numpy as np

m_block_length = 16384;
samples = [0] * 2 * m_block_length;

while True:
    start = time.time()
    line = sys.stdin.read(2*m_block_length)
    end = time.time()
    print("Reading time: ", end - start)
#
# for line in sys.stdin:
#     #sys.stdout.write(line)
    start = time.time()
    nums = array.array('B', line)
    # print len(nums.tolist())

    for i in range(0, m_block_length):
        re = nums[2*i];
        im = nums[2*i+1];
        samples[i] = ( (re - 128) / 128.0) + 1j*((im - 128) / 128.0);

    #print samples
    x1 = np.array(samples).astype("complex64")
    end = time.time()
    print("Parse time: ", end - start)
