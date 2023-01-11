import torch
import torch.nn as nn
from torchsummary import summary
from models.mobilenetv2 import mobilenet_v2
from models.mobilenetv3 import mobilenet_v3
from models.mobilenetv1 import mobilenet_v1

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        # self.stage_repeats = [2, 4, 2]
        # self.stage_repeats = stage_repeats
        # self.stage_out_channels = [-1, 24, 48, 96, 192]
        # self.stage_out_channels = stage_out_channels
        # self.backbone = mobilenet_v2()
        self.backbone = mobilenet_v1()
        # self.backbone = mobilenet_v3()

        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # # self.SPP = SPPBottleneck(sum([40,112,160]), 112)
        # self.SPP = SPP(sum([40, 112, 160]), 112)
        # self.detect_head = DetectHead(112, category_num)
        # self.SPP = SPPBottleneck(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
        # self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
        # self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)

    def forward(self, x):
        # outputs = []
        # x1,x2,P1, P2, P3 = self.backbone(x)
        P1, P2, P3 = self.backbone(x)
        # outputs.append(x1)
        # outputs.append(x2)
        # outputs.append(P1)
        # outputs.append(P2)
        # outputs.append(P3)
        # P3 = self.upsample(P3)
        # P1 = self.avg_pool(P1)

        # P = torch.cat((P1, P2, P3), dim=1)
        # outputs.append(P)
        # y = self.SPP(P)
        # outputs.append(y)
        # head = self.detect_head(y)
        # outputs.append(head)
        # return head,outputs
        return P1, P2, P3

if __name__ == '__main__':
    test_data = torch.rand(1, 3, 320, 320)
    model = Detector()
    p1,p2,p3 = model.forward(test_data)
    print(p1.shape)
    summary(model, input_size=(3, 320, 320), batch_size=1, device='cpu')
    # prin`t(model)
    # print(model.backbone)