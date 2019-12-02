import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from faceboxes import FaceBoxes               # 导入模型


dummy_input = torch.rand(1, 3, 1024, 1024)     # 假设输入13张1*28*28的图片
model = FaceBoxes()
with SummaryWriter(comment='FaceBoxes') as w:
    w.add_graph(model, (dummy_input, ))
