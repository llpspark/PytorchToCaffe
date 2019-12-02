import torch
import numpy as np
from torch.autograd import Variable

# from model.faceboxes import build_net
# from model.configs.CC import Config
from src_model.faceboxes import FaceBoxes

import pytorch_to_caffe

# cfg = Config.fromfile("model/configs/Pelee_PCD.py")

# net = build_net('test', cfg.model.input_size, cfg.model)
net = FaceBoxes(phase='test', size=1024, num_classes=2)

net.load_state_dict(torch.load("./src_model/FaceBoxesProd.pth",map_location='cpu'))

net.eval()

#input_var = Variable(torch.rand(1, 3, 304, 304))
input_var = Variable(torch.rand(1, 3, 1024, 1024))

# pytorch_to_caffe.trans_net(net,input_var,'Pelee')
pytorch_to_caffe.trans_net(net,input_var,'Faceboxes')
pytorch_to_caffe.save_prototxt('./output/Faceboxes_infer.prototxt')
pytorch_to_caffe.save_caffemodel('./output/Faceboxes_infer.caffemodel')

