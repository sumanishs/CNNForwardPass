#!/local/workspace/tools/anaconda2/bin/python2.7
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
import cv2
os.environ["GLOG_minloglevel"] = "1"
import caffe

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

class input_misc:
    def __init__(self, ist = "NONE", isp = "NONE", wst = "NONE", wsp = "NONE", protxt = "NONE", data_format = "NONE", qm = 0, qn = 0):
        self.ist = ist
        self.isp = isp
        self.wst = wst
        self.wsp = wsp
        self.protxt = protxt
        self.data_format = data_format
        self.qm = qm
        self.qn = qn

    def __str__(self):
        return "Input source type:%s, Input source path:%s \nWeight source type:%s, Weight source path:%s" % (self.ist, self.isp, self.wst, self.wsp)


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
        return "id:%d, name:%s, type:%s, kernel:%d, stride:%d, padding:%s, ic:%d, ih:%d, iw:%d, oc:%d, oh:%d, ow:%d, weight filler:%s, weight filler start:%d" % (self.id, self.layer_name, self.type, self.k, self.s, self.p, self.ic, self.ih, self.iw, self.oc, self.oh, self.ow, self.wtf, self.wfs)

class Processor:

    def compute_parameters(self, graph):
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
    
    def create_graph(self, network):
        graph = []
        
        for layer in network:
            id = int(layer['id'])
            layer_name = layer['layer_name']
            type = layer['type']
    
            if id == 0 and type != 'data':
                print "First layer must be \'data\' layer..."
                assert 0
    
            ic = oc = ih = iw = k = -1
            s = 1
            p = "NONE"
            wtf = "NONE"
            wfs = 1
            ist = isp = wst = wsp = protxt = "NONE"
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
            
            df = "NONE"
            qm = 0
            qn = 0
            if layer.get('input_source_type'):
                ist = layer['input_source_type']     
            if layer.get('input_source_path'):
                isp = layer['input_source_path']     
            if layer.get('weight_source_type'):
                wst = layer['weight_source_type']     
            if layer.get('weight_source_path'):
                wsp = layer['weight_source_path']     
            if layer.get('prototxt'):
                protxt = layer['prototxt']
            if layer.get('data_format'):
                df = layer['data_format']
            if layer.get('qm'):
                qm = layer['qm']
            if layer.get('qn'):
                qn = layer['qn']
            
            if type == "data":
                input_m = input_misc(ist=ist, isp=isp, wst=wst, wsp=wsp, protxt=protxt, data_format=df, qm=qm, qn=qn)
    
        return graph, input_m 

    def find_id(self, graph, name):
        for i in range(len(graph)):
            if graph[i].layer_name == name:
                return graph[i].id
        return -1        
    
    def load_wt_from_caffemodel(self, graph):
        print "Loading prototxt:", self.__input_m.protxt
        print "Loading weights:", self.__input_m.wsp
        weights = {}
        #net = caffe.Net(str(self.__input_m.protxt), 1, weights=str(self.__input_m.wsp))
        net = caffe.Net(str(self.__input_m.protxt), caffe.TEST, weights=str(self.__input_m.wsp))
        for param_name in net.params.keys():
                weight = net.params[param_name][0].data
                print "Param name:", param_name, ", Weight shape:", weight.shape, ", ID:", self.find_id(graph, param_name)
                if self.__input_m.data_format == 'qmn' or self.__input_m.data_format == 'QMN':
                    weight = weight * (2**self.__input_m.qn)
                    weight = weight.astype('int64')
                layer_id = self.find_id(graph, param_name)
                if layer_id != -1:
                    weights[layer_id] = weight
                else:
                    assert 0
        return weights
        
    
    def create_weight_pool(self, graph, M, N):
        print "Creating weight pool..."
        weight_pool = {}
        
        if self.__input_m.wst == 'caffemodel':
            print "Loading weights from <", self.__input_m.wsp, ">"
            weight_pool = self.load_wt_from_caffemodel(graph)
        elif self.__input_m.wst == 'dummy':
            print "Creating dummy weights...."
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
    
    def create_input_data(self, inode):
        data = np.zeros([inode.ic, inode.oh, inode.ow], dtype = np.float64)   #Image size 10 x 10 x 4
        if self.__input_m.ist == 'image_file':
            print "Loading input data from <", self.__input_m.isp, ">"
            img = cv2.imread(self.__input_m.isp, 0)
            print "Img shape:", img.shape
            if img.shape != [inode.ih, inode.iw]:
                img2 = cv2.resize(img,(inode.ih, inode.iw))
                img = img2.reshape(inode.ih, inode.iw);
            else:
                img = img.reshape(inode.ih, inode.iw);
            print "Img shape:", img.shape
            img = img/255.0
            print type(img)
            for i in range(inode.ic):
                data[i] = img
        elif self.__input_m.ist == 'dummy':    
            for i in range(inode.ic):
                cl.fillarray_2d(data[i], inode.wtf, i)

        if self.__input_m.data_format == 'qmn' or self.__input_m.data_format == 'QMN':
            data = data * (2**self.__input_m.qn)
            data = data.astype('int64')
        return data

    def process(self, args):    
        js_file = args.net_json
        M = int(args.row)
        N = int(args.col)
        print "Net file:", js_file
        with open(js_file) as f:
            network = json.load(f)

        graph, self.__input_m = self.create_graph(network)
        print self.__input_m
        self.compute_parameters(graph)
        for l in graph:
            print l
        
        inode = graph[0]
        if inode.id != 0 and inode.type != 'data' :
            print "Data layer not found. First layer must be data layer."
            assert 0
        data = self.create_input_data(inode)
        print "Input data shape:", data.shape
        
        
        weight_pool = self.create_weight_pool(graph, M, N)
        for key, value in weight_pool.items(): 
            print "Layer:", key
            print "Weight matrix shape:", value.shape
            print "Matrix:\n", value     
        #return
        
        output_stack = []
        
        output_stack.append(data)
        nnode = len(graph)

        for i in range(1, nnode):
            l = graph[i] 
            input = output_stack.pop()
            if self.__input_m.data_format == 'qmn' or self.__input_m.data_format == 'QMN':
                input = input.astype('int64')
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
                    input = cl.flatten(input)
                    #input = flat
                 weights = weight_pool[l.id] 
                 output = cl.fully_connected_1d(input, weights, l.oc) 
            elif l.type == 'SOFTMAX':
                 print "SoftMax Input:", input
                 output = cl.softmax(input)
                 print "SoftMax Output:", output 
                 for i in range(output.shape[0]):
                    print ("%f" % (output[i]))
            output_stack.append(output)

def main(argv):
    print "Main..."
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j','--net_json', default=1, help='NN json file', required=True)
    parser.add_argument('-m','--row', default=4, help='Systolic array rows', required=False)
    parser.add_argument('-n','--col', default=4, help='Systolic array columns', required=False)
    args = parser.parse_args()

    task = Processor()
    task.process(args)

         
if __name__ == "__main__":
    main(sys.argv)
