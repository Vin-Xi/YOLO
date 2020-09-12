from __future__ import division

import torch
from torch.autograd import Variable
import numpy as np
from util import *
import cv2


def get_test_input():
    img = cv2.imread("cfg/dog.jpg")
    print(img.shape)
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_config(cfg):
    #Opening the file read only mode.
    file=open(cfg,'r') 
    #Store lines in a list
    lines=file.read().split('\n')
    #Remove empty lines
    lines=[line for line in lines if len(line)>0]
    #Remove comments
    lines=[line for line in lines if line[0]!='#']
    #Trimming trailing whitespaces
    lines=[line.rstrip().lstrip() for line in lines]


    block={}
    blocks=[]

    for line in lines:
        if line[0]=="[":
            if len(block)!=0:
                blocks.append(block)
                block={}
            block["type"]=line[1:-1].rstrip()
        else:
            key,value=line.split("=")
            block[key.rstrip()]=value.lstrip()
    blocks.append(block)

    return blocks


class DetectionLayer(torch.nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors=anchors

class EmptyLayer(torch.nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()

def create_layers(blocks):
    net=blocks[0]
    module_list=torch.nn.ModuleList()
    prev_filters=3
    output_filter=[]
    
    for idx,block in enumerate(blocks[1:]):
        module=torch.nn.Sequential()
        
        if (block["type"]=="convolutional"):
            activation=block["activation"]
            print(idx)
            try:
                batch_normalize=int(block["batch_normalize"])
                bias=False
            except:
                batch_normalize=0
                bias=True
            filters=int(block["filters"])
            padding=int(block["pad"])
            kernel_size=int(block["size"])
            stride=int(block["stride"])

            if padding:
                pad=(kernel_size-1)//2

            else:
                pad=0

            conv=torch.nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(idx),conv)

            if batch_normalize:
                bn=torch.nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(idx),bn)

            if activation=="leaky":
                actvn=torch.nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(idx),actvn)
        elif (block["type"]=="upsample"):
            stride=int(block["stride"])
            upsample=torch.nn.Upsample(scale_factor=2,mode="bilinear")
            module.add_module("upsample_{}".format(idx),upsample)
        elif (block["type"]=="route"):
            block["layers"]=block["layers"].split(",")
            start=int(block["layers"][0])

            try:
                end=int(block["layers"][1])
            except:
                end=0

            if start>0:
                start=start-idx
            if end>0:
                end=end-idx
            route=EmptyLayer()
            module.add_module("route_{0}".format(idx),route)
            if end<0:
                filters=output_filter[idx+start]+output_filter[idx+end]
            else:
                filters=output_filter[idx+start]

        elif (block["type"]=="shortcut"):
            shortcut=EmptyLayer()
            module.add_module("shortcut_{}".format(idx),shortcut)

        elif (block["type"]=="yolo"):
            masks=block["mask"].split(",")
            masks=[int(mask) for mask in masks]
            anchors=block["anchors"].split(",")
            anchors=[int(anchor) for anchor in anchors]
            anchors=[(anchors[i],anchors[i+1]) for i in range (0,len(anchors),2)]
            anchors=[anchors[i] for i in masks]
            detection=DetectionLayer(anchors)
            module.add_module("Detection_{}".format(idx),detection)

        module_list.append(module)
        prev_filters=filters
        output_filter.append(filters)
    return (net,module_list)



class Darknet(torch.nn.Module):
    def __init__(self,cfg):
        super(Darknet,self).__init__()
        self.blocks=parse_config(cfg)
        self.net,self.module_list=create_layers(self.blocks)



    def forward(self,x,CUDA=False):
        modules=self.blocks[1:]
        outputs={}

        write=0
        for i,module in enumerate(modules):
            module_type=(module["type"])
            if module_type=="convolutional" or module_type=="upsample":
                x=self.module_list[i](x)
            elif module_type=="route":
                layers=module["layers"]
                layers=[int(layer) for layer in layers]
                if(layers[0]>0):
                    layers[0]=layers[0]-i
                if len(layers)==1:
                    x=outputs[i+(layers[0])]
                else:
                    if (layers[1])>0:
                        layers[1]=layers[1]-i

                    map1=outputs[i+layers[0]]
                    map2=outputs[i+layers[1]]
                    x=torch.cat((map1,map2),1)
            elif module_type=="shortcut":
                from_=int(module["from"])
                x=outputs[i-1]+outputs[i+from_]
            elif module_type=="yolo":
                anchors=self.module_list[i][0].anchors

                inp_dim=int(self.net["height"])
                num_classes=int (module["classes"])

                x=x.data
                print(x.shape)
                print(inp_dim)
                print(num_classes)
                x=predict_transform(x,inp_dim,anchors,num_classes,CUDA)
                if not write:
                    detections=x
                    write=1
                else:
                    detections=torch.cat((detections,x),1)
            outputs[i]=x
        try:
            return detections
        except:
            return 0


    def load_weights(self,wfile):
        fp=open(wfile,"rb")
        header=np.fromfile(fp,dtype=np.int32,count=5)
        self.header=torch.from_numpy(header)
        self.seen=self.header[3]
        weights=np.fromfile(fp,dtype=np.float32)
        ptr=0
        for i in range(len(self.module_list)):
            module_type=self.blocks[i+1]["type"]
            if module_type=="convolutional":
                model=self.module_list[i]
                try:
                    batch_normal=int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normal=0
                conv=model[0]
                if batch_normal:
                    bn=model[1]
                    num_bn_biases=bn.bias.numel()
                    bn_biases=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_weights=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_running_mean=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_running_var=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases=conv.bias.numel()
                    conv_biases=torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr+=num_biases
                    conv_biases=conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)


                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
