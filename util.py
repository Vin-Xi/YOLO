from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]


    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    print(prediction.shape)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
   
    
    return prediction

def write_results(prediction,confidence,num_classes,nms_conf=0.4):
    conf_mask=(prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction=prediction*conf_mask
    corners=prediction.new(prediction.shape)
    #the order follows as centre x, centre y , width , height
    #Converting to x1,y1 and x2,y2
    corners[:,:,0]=(prediction[:,:,0]-prediction[:,:,2]/2)
    corners[:,:,1]=(prediction[:,:,1]-prediction[:,:,3]/2)
    corners[:,:,2]=(prediction[:,:,0]+prediction[:,:,2]/2)
    corners[:,:,3]=(prediction[:,:,1]+prediction[:,:,3]/2)
    prediction[:,:,:4]=corners[:,:,:4]

    write=False

    for i in range(prediction.size(0)):
        image_pred=prediction[i]
        max_conf,max_conf_score=torch.max(image_pred[:,5:5+num_classes],1)
        max_conf=max_conf.float().unsqueeze(1)
        max_conf_score=max_conf_score.float().unsqueeze(1)
        temp=(image_pred[:,:5],max_conf,max_conf_score)
        image_pred=torch.cat(temp,1)

        zero_complement=(torch.nonzero(image_pred[:,4]))
        try:
            image_pred_=image_pred[zero_complement.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        get_classes=unique(image_pred[:,-1])

        for class_ in get_classes:
            class_mask=image_pred_*(image_pred_[:,-1]==class_).float().unsqueeze(1)
            class_mask_ind=torch.nonzero(class_mask[:,-2]).squeeze()
            image_pred_class=image_pred_[class_mask_ind].view(-1,7)

            sort_conf=torch.sort(image_pred_class[:,4],descending=True)[1]
            image_pred_class=image_pred_class[sort_conf]
            idx=image_pred_class.size(0)

            for j in range(idx):
                try:
                    ious=bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                iou_mask=(ious>nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:]*=iou_mask
                non_zero_ind=torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class=image_pred_class[non_zero_ind].view(-1,7)

            batch_ind=image_pred_class.new(image_pred_class.size(0),1).fill_(i)
            temp=batch_ind,image_pred_class
            if not write:
                output=torch.cat(temp,1)
                write=True
            else:
                out=torch.cat(temp,1)
                output=torch.cat((output,out))

    try:
        return output
    except:
        return 0



def unique(tensor):
    tensor_np=tensor.cpu().numpy()
    unique_np=np.unique(tensor_np)
    unique_tensor=torch.from_numpy(unique_np)
    tensor_res=tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1,box2):
    b1x1,b1y1,b1x2,b1y2=box1[:,0],box1[:,1],box1[:,2],box1[:,3]
    b2x1,b2y1,b2x2,b2y2=box2[:,0],box2[:,1],box2[:,2],box2[:,3]

    intersection_x1=torch.max(b1x1,b2x1)
    intersection_y1=torch.max(b1y1,b2y1)
    intersection_x2=torch.min(b1x2,b2x2)
    intersection_y2=torch.min(b2y2,b2y2)

    intersection_area=torch.clamp(intersection_x2-intersection_x1+1,min=0)*torch.clamp(intersection_y2-intersection_y1+1,min=0)

    box1_area=(b1x2-b1x1+1)*(b1y2-b1y1+1)
    box2_area=(b2x2-b2x1+1)*(b2y2-b2y1+1)

    iou=intersection_area/(box1_area+box2_area-intersection_area)
    return iou


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas