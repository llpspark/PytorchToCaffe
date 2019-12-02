import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ONet(nn.Module):
    def __init__(self, test=True):
        super(ONet, self).__init__()

        self.test = test
        if test:
            self.softmax = nn.Softmax(dim=1)

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )

        self.fc = nn.Linear(128 * 3 * 3, 256)
        self.relu = nn.PReLU()

        self.cls = nn.Linear(256, 2)
        self.reg = nn.Linear(256, 4)

    def forward(self, x):
        x = self.extractor(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.relu(self.fc(x))
        cls = self.cls(x)
        reg = self.reg(x)
        if self.test:
            cls = self.softmax(cls)
        return cls, reg 



if __name__ == '__main__':
    net = ONet(test=True)
    state_dict = torch.load('/home/liangfeng/grocery/model_convert/PytorchToCaffe-master/src_model/onet_20181218_final.pkl')
    new_state_dict = OrderedDict()
    state_dict = state_dict['weights']
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
        # print(k)
        # print(v)
    # torch.save(new_state_dict, './onet_new.pth')
    net.load_state_dict(new_state_dict)

    for name, layer in net.named_modules():
        print("name: ", name)
        print("layer: ", layer)
    inputs = torch.randn(1, 3, 48, 48)
    out = net(inputs)
    # print(out[0])
    # print(out[1])

