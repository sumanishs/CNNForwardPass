#!/local/workspace/tools/anaconda2/bin/python2.7
####################################################
####  Dummy Weight generation script for DW2 #######
####  sumanish.sarkar@sasken.com  
####################################################
import os
import struct
import sys
import math
import numpy as np
import argparse
import random

## format weight data in required MxN systolic array format for dw2
def genWtDataConv(sysarray, wt_list, tfd,tfh):
    OPch = len(wt_list)    
    M = sysarray.shape[0]
    N = sysarray.shape[1]
    K = wt_list[0].shape[1]
    IPch = wt_list[0].shape[0]
    outseti = OPch/M
    if OPch % M > 0:
        outseti = outseti + 1
    inseti = IPch/N
    if IPch % M > 0:
        inseti = inseti + 1
    for oc in range(outseti):
        for ic in range(inseti):
            for x in range(K):
                for y in range(K):
                    for m in range(M):
                        ocdx = oc*M + m
                        for n in range(N):
                            icdx = ic*N + n
                            if ocdx < len(wt_list):
                                mat = wt_list[ocdx]
                                if icdx < mat.shape[0]:
                                    layermat = mat[icdx]
                                    val = layermat[x][y]
                                    sysarray[m][n] = val
                    t = np.transpose(sysarray)
                    c = np.reshape(t, M * N)
                    for l in range(M * N):
                        tfd.write("%d\n" % c[l])
                        tfh.write("%X\n" % c[l])
                    sysarray.fill(0) 
                

def fillarray(array, filler, startval, arrN):
    mval = 0	## used in mod filter 
    val = startval ## used in other filters 
    if filler == 'mod':
        for i in range(array.shape[0]):
            f = mval % arrN
            array[i].fill(f+1)
            mval = mval+1
    elif filler == 'inc':
	print "incremental filling of weight values.....\n"
	## fills the 5x5 array with incremental values from the startvalue 
	for i in range(array.shape[0]):	
	    for j in range(array.shape[1]):	
	        for k in range(array.shape[2]):
            	    array[i,j,k]=(val)
        	    val = val + 1

    elif filler == 'rowinc':
	print "row increment and repeat for all rows....\n"
	## fills the 5x5 array with incremental values for an row . then repeat from startvalue again  
	for i in range(array.shape[0]):	
	    for j in range(array.shape[1]):	
		val=startval		## repeat for the rows 
	        for k in range(array.shape[2]):
            	    array[i,j,k]=(val)
        	    val = val + 1
    elif filler == 'const':
        for i in range(array.shape[0]):
            array[i].fill(val)
            val = val + 1
    elif filler == 'rand':
        for i in range(array.shape[0]):
            array[i].fill(random.randint(0, 3))
    else:
        print "Wrong filler type.."
        exit()

def fillarray_fc(array, filler, startval, arrN):
    if filler == 'inc':
        for i in range(arrN):
            array[i] = startval + i   
    elif filler == 'rand':
        array.fill(random.randint(0, 3))    
        
def genWtDataFC0(sysarray, wt_list, tfd, tfh):
    OPch = len(wt_list)    
    M = sysarray.shape[0]
    N = sysarray.shape[1]
    IPch = wt_list[0].shape[0]
    chunks = OPch/M
    if OPch % M > 0:
        chunks += 1
    for i in range(chunks):
        for e in range(IPch):
           for m in range(M):
                if i * M + m < OPch:
                    arr = wt_list[i * M + m]
                    val = arr[e]
                    sysarray[m][0] = val
           t = np.transpose(sysarray)
           c = np.reshape(t, M * N)
           for l in range(M * N):
               tfd.write("%d\n" % c[l])
               tfh.write("%X\n" % c[l])
           sysarray.fill(0) 

def genWtDataFC1(sysarray, wt_list, tfd, tfh):
    OPch = len(wt_list)
    M = sysarray.shape[0]
    N = sysarray.shape[1]
    IPch = wt_list[0].shape[0]
    ichunks = OPch/N
    if OPch % N > 0:
        ichunks += 1
    ochunks = IPch/M
    if IPch % M > 0:
        ochunks += 1
    e = 0
    for i in range(ichunks):
        for o in range(ochunks):
                for m in range(M):
                    for n in range(N):
                        if i * N + n < OPch:
                            arr = wt_list[i * N + n]
                            if e < IPch:
                                val = arr[e]
                                sysarray[m][n] = val
                    e += 1
                    if e >= IPch:
                        e = 0
                        break;
                #print sysarray
                #t = np.transpose(sysarray)
                #print "Transposed:\n", t
                c = np.reshape(sysarray, M * N)
                #print "Reshaped:\n", c
                for l in range(M * N):
                    tfd.write("%d\n" % c[l])
                    tfh.write("%X\n" % c[l])
                sysarray.fill(0)
                
