import torch
from torch import nn
from torchviz import make_dot

from faceboxes import FaceBoxes

model = FaceBoxes()

x = torch.randn(1, 3, 1024, 1024).requires_grad_(True)
y = model(x)
vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
vis_graph.view()
