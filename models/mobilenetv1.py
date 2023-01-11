import torch
import torch.nn as nn
from torchsummary import summary

def conv_bn(in_channel, out_channel, stride=1):
    """
        传统卷积块：Conv+BN+Act
    """
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )


def conv_dsc(in_channel, out_channel, stride=1):
    """
        深度可分离卷积：DW+BN+Act + Conv+BN+Act
    """
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
        nn.BatchNorm2d(in_channel),
        nn.ReLU6(inplace=True),

        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self, in_dim=3, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes
        self.stage1 = nn.Sequential(

            conv_bn(in_dim, 32, 2),
            conv_dsc(32, 64, 1),

            conv_dsc(64, 128, 2),
            conv_dsc(128, 128, 1),

            conv_dsc(128, 256, 2),
            conv_dsc(256, 256, 1),
        )

        self.stage2 = nn.Sequential(
            conv_dsc(256, 512, 2),
            conv_dsc(512, 512, 1),
            conv_dsc(512, 512, 1),
            conv_dsc(512, 512, 1),
            conv_dsc(512, 512, 1),
            conv_dsc(512, 512, 1),
        )

        self.stage3 = nn.Sequential(
            conv_dsc(512, 1024, 2),
            conv_dsc(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(1024, self.num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class mobilenet_v1(nn.Module):
    def __init__(self):
        super(mobilenet_v1, self).__init__()
        self.model = MobileNetV1()

    def forward(self, x):
        out3 = self.model.stage1(x)
        out4 = self.model.stage2(out3)
        out5 = self.model.stage3(out4)
        return out3, out4, out5

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1().to(device)
    summary(model, input_size=(3, 320, 320))
