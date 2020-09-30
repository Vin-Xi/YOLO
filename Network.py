import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from pser import parser

#Shortcut Layer to add outputs of previous layers
class Shortcut(nn.Module):
    def __init__(self, index):
        super(Shortcut, self).__init__()
        self.index = index

    def forward(self, x, outputs):
        return x + outputs[self.index]
#Provides a jump to add into next layers
class Route(nn.Module):
    def __init__(self, indice):
        super(Route, self).__init__()
        self.indice = indice

    def forward(self, output):
        out_stream = [output[i] for i in self.indice]
        out_stream = torch.cat(out_stream, dim=1)
        return out_stream

#There are 3 YOLO layers and this class will convert them into YOLO format [center_x, center_y, width, height, objectness score, class scores...]
class YOLOLayer(nn.Module):
    def __init__(self, anch, n_C, inp_dim):
        super(YOLOLayer, self).__init__()
        self.anch = torch.tensor(anch, dtype=torch.float)
        self.n_C = n_C
        self.n_A = len(anch)
        self.inp_dim = inp_dim

    def forward(self, x):
        batch_size = x.size(0)
        n_G = x.size(2)
        stride = self.inp_dim // n_G
        #Reshaping the input with shape [batch_size,number_anchors,classes+5,grid_size,grid_size]
        detection = x.view(batch_size, self.n_A, self.n_C + 5, n_G, n_G)
        # Applying a sigmoid function on box centers to bring them between 1 and 0
        detection[:, :, :2, :, :] = torch.sigmoid(detection[:, :, :2, :, :])
        # objectness score and class scores
        detection[:, :, 4:, :, :] = torch.sigmoid(detection[:, :, 4:, :, :])

        #Adding Offset to the centers

        x_offset, y_offset = np.meshgrid(np.arange(n_G), np.arange(n_G), indexing='xy')
        x_offset = torch.from_numpy(x_offset).float()
        y_offset = torch.from_numpy(y_offset).float()

        
        
        x_offset = x_offset.expand_as(detection[:, :, 0, :, :])
        y_offset = y_offset.expand_as(detection[:, :, 1, :, :])
        detection[:, :, 0, :, :] += x_offset
        detection[:, :, 1, :, :] += y_offset
        # rescale to original image dimention
        detection[:, :, :2, :, :] *= stride

        # box width and height
        anch = self.anch.unsqueeze(-1).unsqueeze(-1).expand_as(detection[:, :, 2:4, :, :])
        
        detection[:, :, 2:4, :, :] = torch.exp(detection[:, :, 2:4, :, :]) * anch
        detection = detection.transpose(1, 2).contiguous().view(batch_size, self.n_C+5, -1).transpose(1, 2)

        return detection

def create_modules(blocks):
    net_info = blocks[0]    # the first block is network info
    module_list = nn.ModuleList()
    in_channel = 3
    out_channel = in_channel
    out_channels = []   # keep track of output channel for every block for specifying conv layer input channels

    for i, block in enumerate(blocks[1:]):
        block_type = block['type']
        if block_type == 'convolutional':
            module = nn.Sequential()
            if 'batch_normalize' in block.keys():
                bn = True
                bias = False
            else:
                bn = False
                bias = True
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            pad = int(block['pad'])
            activation = block['activation']

            if pad:
                padding = (kernel_size-1) // 2
            else:
                padding = 0

            conv = nn.Conv2d(in_channels=in_channel, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            module.add_module('conv_%d' % (i), conv)

            if bn:
                module.add_module('batchnorm_%d' %(i), nn.BatchNorm2d(filters))
            if activation == 'leaky':
                module.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))

            out_channel = filters

        elif block_type == 'shortcut':
            idx = int(block['from']) + i
            module = Shortcut(idx)

        elif block_type == 'upsample':
            stride = int(block['stride'])
            module = nn.Upsample(scale_factor=stride, mode='bilinear')

        # route block could have one or two indices. Negative value means relative index.
        elif block_type == 'route':
            layer_indices = block['layers'].split(',')
            first_idx = int(layer_indices[0])
            if first_idx < 0:
                first_idx = i + first_idx
            if len(layer_indices) > 1:
                second_idx = int(layer_indices[1])
                if second_idx < 0:
                    second_idx += i
                out_channel = out_channels[first_idx] + out_channels[second_idx]
                module = Route([first_idx, second_idx])
            else:
                out_channel = out_channels[first_idx]
                module = Route([first_idx])


        elif block_type == 'yolo':
            masks = block['mask'].split(',')
            masks = [int(mask) for mask in masks]
            anchors = block['anchors'].split(',')
            anchors = [[int(anchors[2*i]), int(anchors[2*i+1])] for i in masks]
            num_classes = int(block['classes'])
            input_dim = int(net_info['width'])
            module = YOLOLayer(anchors, num_classes, input_dim)

        out_channels.append(out_channel)
        in_channel = out_channel
        module_list.append(module)

    return (net_info, module_list)

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.blocks = parser(cfg)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        blocks = self.blocks[1:]
        outputs = []
        detections = torch.tensor([], dtype=torch.float)
        detections = Variable(detections)
        for i, module in enumerate(self.module_list):
            block_type = blocks[i]['type']
            if block_type == 'convolutional' or block_type == 'upsample':
                x = module(x)
            elif block_type == 'shortcut':
                x = module(x, outputs)
            elif block_type == 'route':
                x = module(outputs)
            elif block_type == 'yolo':
                x = module(x)
                detections = torch.cat((x, detections), dim=1)

            outputs.append(x)

        return detections

    '''
    Weights file structure:
    - header: 5 integers
    - weights of conv layers 
        - conv layer with batch_norm: [bn_bias, bn_weight, bn_running_meanm, bn_running_var, conv_weight]
        - conv layer without batch_norm: [conv_bias, conv_weight]
    '''
    def load_weights(self, file):
        with open(file, 'rb') as fi:
            header = np.fromfile(fi, np.int32, count=5)
            weights = np.fromfile(fi, np.float32)
        self.header = torch.from_numpy(header)
        pointer = 0

        for i in range(len(self.module_list)):
            module = self.module_list[i]
            block_type = self.blocks[i+1]['type']

            if block_type == 'convolutional':
                conv = module[0]
                if 'batch_normalize' in self.blocks[i+1].keys():
                    bn = module[1]
                    number_weights = bn.weight.numel()

                    bn_bias = torch.from_numpy(weights[pointer: pointer + number_weights]).view_as(bn.bias.data)
                    pointer += number_weights
                    bn_weight = torch.from_numpy(weights[pointer: pointer + number_weights]).view_as(bn.weight.data)
                    pointer += number_weights

                    bn_running_mean = torch.from_numpy(weights[pointer: pointer + number_weights]).view_as(bn.running_mean)
                    pointer += number_weights
                    bn_running_var = torch.from_numpy(weights[pointer: pointer + number_weights]).view_as(bn.running_var)
                    pointer += number_weights

                    bn.weight.data.copy_(bn_weight)
                    bn.bias.data.copy_(bn_bias)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_bias = conv.bias.numel()
                    conv_bias = torch.from_numpy(weights[pointer: pointer + num_bias]).view_as(conv.bias.data)
                    pointer += num_bias
                    conv.bias.data.copy_(conv_bias)

                number_weights = conv.weight.numel()
                conv_weight = torch.from_numpy(weights[pointer: pointer + number_weights]).view_as(conv.weight.data)
                pointer += number_weights
                conv.weight.data.copy_(conv_weight)

