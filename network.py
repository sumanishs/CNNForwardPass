#!/usr/bin/python2.7
import os
import struct
import sys
import math
import numpy as np
import argparse
import random
import json
import cnn_layers as cl
import weight_formatter as wf

class node:
    def __init__ (self, id = 0, layer_name = "dummy", type = "dummy", ic = -1, ih = -1, iw = -1, oc = -1, k = -1, s = 1, p = "NONE", wtf = "NONE", wfs = 1):
        self.id = id
        self.layer_name = layer_name
        self.type = type
        self.ic = ic
        self.ih = ih
        self.iw = iw
        self.oc = oc
        self.k = k
        self.s = s
        self.p = p
        self.oh = -1
        self.ow = -1
        self.wtf = wtf
        self.wfs = wfs

    def __str__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return "id:%d, name:%s, type:%s, kernel:%d, stride:%d, padding:%s, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, weight filler:%s, weight filler start:%d" % (self.id, self.layer_name, self.type, self.k, self.s, self.p, self.ic, self.ih, self.iw, self.oc, self.oh, self.ow, self.wtf, self.wfs)

def compute_parameters(graph):
    assert len(graph) > 0
    inode = graph[0]
    if inode.id != 0 and inode.type != 'data' :
        print "Data layer not found. First layer must be data layer."
        assert 0

    data_ic = inode.ic
    data_ih = inode.ih
    data_iw = inode.iw
    inode.oh = inode.ih
    inode.ow = inode.iw
    inode.oc = inode.ic
    print "Input data: Input channels:", data_ic, ", Height:", data_iw, ", Width:", data_iw
    #print inode 
    nnode = len(graph)
    for i in range(nnode):
        l = graph[i]
        if l.type == 'data':
            continue
        if l.type == 'CONV':
            prev = graph[i-1]
            l.ic = ic = prev.oc
            l.ih = ih = prev.oh
            l.iw = iw = prev.ow
            k = l.k
            if l.p == 'SAME':
                ih += k-1
                iw += k-1 
            oh = ((ih - k)/l.s) + 1
            ow = ((iw - k)/l.s) + 1
            l.oh = oh
            l.ow = ow
        if l.type == 'POOL':
            prev = graph[i-1]
            l.ic = ic = prev.oc
            l.ih = ih = prev.oh
            l.iw = iw = prev.ow
            k = l.k
            oh = ((ih - k)/l.s) + 1
            ow = ((iw - k)/l.s) + 1
            l.oc = l.ic
            l.oh = oh
            l.ow = ow
        if l.type == 'FC':
            prev = graph[i-1]
            l.ic = ic = prev.oc * prev.oh * prev.ow
            l.ih = 1
            l.iw = 1
            l.oh = 1
            l.ow = 1
        if l.type == 'RELU':     
            prev = graph[i-1]
            l.ic = prev.oc
            l.ih = prev.oh
            l.iw = prev.ow
            l.oc = l.ic
            l.oh = l.ih
            l.ow = l.iw
        if l.type == 'SOFTMAX':     
            prev = graph[i-1]
            l.ic = prev.oc
            l.ih = prev.oh
            l.iw = prev.ow
            l.oc = l.ic
            l.oh = l.ih
            l.ow = l.iw

def create_graph(network):
    graph = []
    for layer in network:
        id = int(layer['id'])
        layer_name = layer['layer_name']
        type = layer['type']
        ic = oc = ih = iw = k = -1
        s = 1
        p = "NONE"
        wtf = "NONE"
        wfs = 1
        if layer.get('input_channel'):
            ic = int(layer['input_channel'])        
        if layer.get('output_num'):
            oc = int(layer['output_num'])        
        if layer.get('input_height'):
            ih = int(layer['input_height'])        
        if layer.get('input_width'):
            iw = int(layer['input_width'])        
        if layer.get('kernel_size'):
            k = int(layer['kernel_size'])        
        if layer.get('stride'):
            s = int(layer['stride'])     
        if layer.get('padding'):
            p = layer['padding']     
        if layer.get('weight_filler'):
            wtf = layer['weight_filler']     
        if layer.get('weight_filler_start'):
            wfs = layer['weight_filler_start']     
        n = node(id = id, layer_name = layer_name, type = type, ic = ic, ih = ih, iw = iw, oc = oc, k = k, s = s, p = p, wtf = wtf, wfs = wfs) 
        graph.append(n)  
    return graph 

def create_weight_pool(graph, M, N):
    print "Creating weight pool..."
    weight_pool = {}
    for l in graph:
        if l.type == 'CONV':
            ic = l.ic
            oc = l.oc
            k = l.k
            wt = np.zeros([oc, ic, k, k], dtype = float) 
            for i in range(oc):
                #cl.fillarray_3d(wt[i], l.wtf)
                startval = l.wfs +(i*1)
                wf.fillarray(wt[i], l.wtf, int(startval), N)			 
            weight_pool[l.id] = wt
        if l.type == 'FC':
            oc = l.oc 
            ic = l.ic    
            wt = np.zeros([oc, ic], dtype = float)
            for i in range(oc):
                #cl.fillarray_1d(wt[i], l.wtf, i)
                startval = l.wfs +(i*1)
                wf.fillarray_fc(wt[i], l.wtf, int(startval),  ic)			
            weight_pool[l.id] = wt
    return weight_pool

def create_input_data(inode):
    data = np.zeros([inode.oc, inode.oh, inode.ow], dtype = float)   #Image size 10 x 10 x 4
    for i in range(inode.oc):
        cl.fillarray_2d(data[i], inode.wtf, i)
    return data

def main(argv):
    print "Main..."
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j','--net_json', default=1, help='NN json file', required=True)
    parser.add_argument('-m','--row', default=4, help='Systolic array rows', required=False)
    parser.add_argument('-n','--col', default=4, help='Systolic array columns', required=False)
    args = parser.parse_args()
    js_file = args.net_json
    M = int(args.row)
    N = int(args.col)
    print "Net file:", js_file
    with open(js_file) as f:
        network = json.load(f)
    #print(network)

    graph = create_graph(network)
    compute_parameters(graph)
    for l in graph:
        print l
    weight_pool = create_weight_pool(graph, M, N)
    for key, value in weight_pool.items(): 
        print "Layer:", key
        print "Weight matrix shape:", value.shape
        print "Matrix:\n", value     
    
    output_stack = []
    
    inode = graph[0]
    if inode.id != 0 and inode.type != 'data' :
        print "Data layer not found. First layer must be data layer."
        assert 0
   
    data = create_input_data(inode)
    output_stack.append(data)
    nnode = len(graph)

    for i in range(1, nnode):
        l = graph[i] 
        input = output_stack.pop()
        print "Input shape:", input.shape
        print "Len:", len(input.shape)
        if l.type == 'CONV':
           weights = weight_pool[l.id] 
           output = cl.conv_op_3d(input, weights, stride = l.s, pad = l.p)   
        elif l.type == 'POOL':
             output = cl.max_pool_3d(input, kernel_size = l.k, stride = l.s)
        elif l.type == 'RELU':
             if len(input.shape) == 1:
                output = cl.ReLu_1d(input)
             else:
                output = cl.ReLu_3d(input)
        elif l.type == 'FC':
             if len(input.shape) != 1:
                flat = cl.flatten(input)
                input = flat
             weights = weight_pool[l.id] 
             output = cl.fully_connected_1d(input, weights, l.oc) 
        elif l.type == 'SOFTMAX':
             print "SoftMax Input:", input
             output = cl.softmax(input)
             print "SoftMax Output:", output  
        output_stack.append(output)

         
if __name__ == "__main__":
    main(sys.argv)
