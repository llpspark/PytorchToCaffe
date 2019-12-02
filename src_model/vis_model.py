import torch
from torch.autograd import Variable
import torch.nn as nn
from graphviz import Digraph


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tensorboardX import SummaryWriter

__all__ = ['view_model']


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        # conv -> bn -> relu
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)

        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)

        return torch.cat([branch1x1, branch1x1_2, branch3x3, branch3x3_3], 1)


class CRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(torch.cat([x, -x], 1), inplace=True)


class FaceBoxes(nn.Module):
    def __init__(self, phase='test', size=1024, num_classes=2):
        super(FaceBoxes, self).__init__()
        self.phase = phase  # 'train' or 'test'
        self.size = size
        self.num_classes = num_classes  # default 2 for face detection, a face or not a face

        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)

        self.inception1, self.inception2, self.inception3 = Inception(), Inception(), Inception()

        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.loc, self.conf = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        if self.phase == 'train':
            # Init weights and biases of current model
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def multibox(self, num_classes):
        loc_layers = [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1),
                      nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1),
                      nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]

        conf_layers = [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1),
                       nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1),
                       nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]

        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):
        detection_sources, loc, conf = list(), list(), list()

        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        x = F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, padding=1)

        # For (1024, 1024) shaped input image, x.shape = torch.Size([1, 128, 32, 32])
        x = self.inception3(self.inception2(self.inception1(x)))
        detection_sources.append(x)

        # For (1024, 1024) shaped input image, x.shape = torch.Size([1, 256, 16, 16])
        x = self.conv3_2(self.conv3_1(x))
        detection_sources.append(x)

        # For (1024, 1024) shaped input image, x.shape = torch.Size([1, 256, 8, 8])
        x = self.conv4_2(self.conv4_1(x))
        detection_sources.append(x)

        for (x, l, c) in zip(detection_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (loc.view(loc.size(0), -1, 4),
                      self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        return output



def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def view_model(net, input_shape):
    x = Variable(torch.randn(1, *input_shape))
    y = net(x)
    y = y[0]
    g = make_dot(y)
    g.view()

    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("layer parameters size:" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("layer parameters:" + str(l))
        k = k + l
    print("total parameters:" + str(k))


if __name__ == '__main__':
    net = FaceBoxes()
    view_model(net,[3, 1024, 1024])