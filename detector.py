
import pickle as pikl
import math
from threading import Thread
from queue import Queue
import random
import os.path as osp
import os
from datetime import datetime
import torch
import cv2 as ocv
import numpy as np
from torch.autograd import Variable
from Network import Network
from util import *

def load_classes(class_):
    fp = open(class_, "r")
    classes = fp.read().split("\n")[:-1]
    return classes

input_="video.mp4"
obj_thresh=0.5
nms_thresh=0.4
output_pipeline="detection/"
video=True
no_show=False

def form_bat(imgs, batches):
    number_bat = math.ceil(len(imgs) // batches)
    batches = [imgs[i*batches : (i+1)*batches] for i in range(number_bat)]

    return batches

def append_bounding(images, bbox, colors, classes):
    img = images[int(bbox[0])]
    annot = classes[int(bbox[-1])]
    first = tuple(bbox[1:3].int())
    second = tuple(bbox[3:5].int())
    color = random.choice(colors)
    ocv.rectangle(img, first, second, color, 2)
    text_size = ocv.getTextSize(annot, ocv.FONT_HERSHEY_COMPLEX, 1, 1)[0]
    third = (first[0], first[1] - text_size[1] - 4)
    fourth = (first[0] + text_size[0] + 4, first[1])
    ocv.rectangle(img, third, fourth, color, -1)
    ocv.putText(img, annot, first, ocv.FONT_HERSHEY_COMPLEX, 1, [235, 255, 255], 1)

def detect_video(model):
    in_size = [int(model.net_info['height']), int(model.net_info['width'])]
    classes = load_classes("data/coco.names")
    colors = pikl.load(open("pallete", "rb"))
    colors = [colors[3]]
    cap = ocv.VideoCapture(input_)
    output_path = osp.join(output_pipeline, 'detection_' + osp.basename(input_).rsplit('.')[0] + '.avi')
    w, h = int(cap.get(ocv.CAP_PROP_FRAME_WIDTH)), int(cap.get(ocv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(ocv.CAP_PROP_FPS)
    fc = ocv.VideoWriter_fourcc(*'XVID')
    out = ocv.VideoWriter(output_path, fc, fps, (w, h))
    read_frames = 0
    init_time = datetime.now()
    print('Detection Started...')
    while cap.isOpened():
        escflag, window = cap.read()
        read_frames += 1
        if escflag:
            f_tensor = cv2_tensor(window, in_size).unsqueeze(0)
            f_tensor = Variable(f_tensor)
            detections = model(f_tensor)
            detections = make_pred(detections, obj_thresh, nms_thresh)
            if len(detections) != 0:
                detections = transform_prediction(detections, [window], in_size)
                for ous in detections:
                    append_bounding([window], ous, colors, classes)

            if not no_show:
                ocv.imshow('window', window)
            out.write(window)
            if not no_show and ocv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    de_time = datetime.now()
    print('Detection Complete in %s' % (de_time - init_time))
    
    cap.release()
    out.release()
    if not no_show:    
        ocv.destroyAllWindows()

    print('Detected video saved to ' + output_path)

    return





if __name__ == '__main__':
    

    print('Initializing Neural Network')
    model = Network("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    model.eval()
    print('Network loaded')
    detect_video(model)

