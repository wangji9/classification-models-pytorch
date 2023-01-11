import os
import json

import torch
from PIL import Image
from datasets import ClassificationDataset, load_iamge, data_transforms
import matplotlib.pyplot as plt

from models.mobilenetv1 import MobileNetV1
from models.mobilenetv3 import MobileNetV3
from models.mobilenetv2 import MobileNetV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Classification_Test():
    def __init__(self) -> None:
        self.test_path = 'Oxford_102_Flowers/data/oxford-102-flowers/valid.txt'
        self.datadir_path = 'Oxford_102_Flowers/data/oxford-102-flowers'
        self.model = MobileNetV2(num_classes=102).to(device)
        self.aug = data_transforms('val')

        fp = open(self.test_path, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(self.datadir_path + '/' + information[0])
            labels.append(int(information[1]))
        self.images = images
        self.labels = labels

    def test(self):
        predict_cla_list = []
        a = 0
        for i, iamge in enumerate(self.images):
            iamge = load_iamge(iamge)
            iamge = self.aug(iamge)
            img = torch.unsqueeze(iamge, dim=0)
            weights_path = "model.pth"
            assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(self.model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                predict_cla_list.append(predict_cla)
            print(
                "class: {:3},label: {:3},prob: {:.3}".format(predict_cla, self.labels[i], predict[predict_cla].numpy()))
            if self.labels[i] == predict_cla_list[i]:
                a = a + 1

        print("acc:{}".format(a / len(self.images)))


if __name__ == '__main__':
    model = Classification_Test()
    model.test()