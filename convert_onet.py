import torch
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict

# from model.faceboxes import build_net
# from model.configs.CC import Config
from src_model.onet import ONet

import pytorch_to_caffe

# cfg = Config.fromfile("model/configs/Pelee_PCD.py")

# net = build_net('test', cfg.model.input_size, cfg.model)
net = ONet(test=True)

# net.load_state_dict(torch.load("/home/liangfeng/grocery/model_convert/FaceBoxesProd.pth",map_location='cpu'))
state_dict = torch.load('./src_model/onet_20181218_final.pkl')
new_state_dict = OrderedDict()
# print(state_dict)
state_dict = state_dict['weights']
for k, v in state_dict.items():
    new_state_dict[k[7:]] = v
# torch.save(new_state_dict, './onet_new.pth')
net.load_state_dict(new_state_dict)


net.eval()

#input_var = Variable(torch.rand(1, 3, 304, 304))
input_var = Variable(torch.rand(1, 3, 48, 48))

# pytorch_to_caffe.trans_net(net,input_var,'Pelee')
pytorch_to_caffe.trans_net(net,input_var,'ONet')
pytorch_to_caffe.save_prototxt('./output/ONet_infer.prototxt')
pytorch_to_caffe.save_caffemodel('./output/ONet_infer.caffemodel')

