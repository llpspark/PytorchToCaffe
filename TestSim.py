import torch
import numpy as np
import cv2
from torch.autograd import Variable

from model.peleenet import build_net
from model.configs.CC import Config


src = cv2.imread("001763.jpg")
src = cv2.resize(src,(304,304))
src = np.transpose(src,(2,0,1))
src = src.astype(np.float32)
src[0]=src[0]-103.94
src[1]=src[1]-116.78
src[2]=src[2]-123.68

src = src*0.017

src = src[np.newaxis,...]

'''

import sys
sys.path.insert(0,"/home/huolei/ssd/caffe_mo/python")
import caffe


caffe_net = caffe.Net("Pelee.prototxt","Pelee.caffemodel",caffe.TEST)

'''

input = torch.from_numpy(src)

cfg = Config.fromfile("model/configs/Pelee_PCD.py")

net = build_net('test',cfg.model.input_size,cfg.model)

net.load_state_dict(torch.load("Pelee/PCD/Pelee_PCD_size304_epoch4.pth",map_location='cpu'))

net.eval()

output = net(input)

for out in output:
    np_out = out.detach().numpy()
    print(np_out)


