# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 08:27:17 2018

This is a basic implementation of an RMS detector
It uses a single pole filter with no latency or look ahead (IIR)
This is considered as a simple, low accuracy, high efficiency envelope follower

Input: Vector, Fs
Output: RMS Value of envelope as a vector

@author: ibarkai
"""

import array 
import numpy as np
import matplotlib.pyplot as plt

def RMS_env(X, Fs):
    
    #Initialze
    Yn1 = 0
    Alpha = 1 - np.e**(- 1 / (Fs * TC / 1000))
    print 'Alpha :', Alpha
    
    
    for i in ary:
        #Y[i] = np.square(X[i]**2*Alpha + Yn1**2*(1-Alpha))
        Y[i] = np.sqrt( X[i]*X[i]*Alpha  + (1-Alpha)*Yn1*Yn1 )
        Yn1 = Y[i]    
    
    return (Y)