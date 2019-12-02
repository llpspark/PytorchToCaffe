import torch
import numpy as np
from torch.autograd import Variable

from model.peleenet import build_net
from model.configs.CC import Config

import pytorch_to_caffe

cfg = Config.fromfile("model/configs/Pelee_PCD.py")

net = build_net('test',cfg.model.input_size,cfg.model)

net.load_state_dict(torch.load("Pelee/PCD/Final_Pelee_PCD_size304.pth",map_location='cpu'))

net.eval()

input_var = Variable(torch.rand(1, 3, 304, 304))

pytorch_to_caffe.trans_net(net,input_var,'Pelee')
pytorch_to_caffe.save_prototxt('Pelee_infer.prototxt')
pytorch_to_caffe.save_caffemodel('Pelee_infer.caffemodel')

