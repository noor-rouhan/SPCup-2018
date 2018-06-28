# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:49:06 2018

@author: User

"""
import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32, float32)"], target ='cpu')
def VectorAdd(a,b):
    return a+ b
 #   return a + b
def main():
    N = 320000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)
    
    start = timer()
    C = VectorAdd(A,B) 
    vectoradd_time = timer() - start
    
    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))
    
    print("%f seconds" % vectoradd_time)
 
    
if __name__ == '__main__':
    main()
    #export NUMBAPRO_NVVM=/usr/local/cuda-8.0/nvvm/lib64/libnvvm.so
    #export NUMBAPRO_LIBDEVICE=/usr/local/cuda-8.0/nvvm/libdevice/ 