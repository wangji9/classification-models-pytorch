import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets import ClassificationDataset,load_iamge,data_transforms
import warnings
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from models.mobilenetv1 import MobileNetV1
from models.mobilenetv3 import MobileNetV3
from models.mobilenetv2 import MobileNetV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net = MobileNetV1(num_classes=102)
# net = MobileNetV2()
# net = MobileNetV3()

class Classification():
    def __init__(self):
        self.train_data_dir = r'D:\xunleidownload\Oxford_102_Flowers\data\oxford-102-flowers\train.txt'
        self.val_data_dir = r'D:\xunleidownload\Oxford_102_Flowers\data\oxford-102-flowers\valid.txt'
        self.datadir_path = r'D:\xunleidownload\Oxford_102_Flowers\data\oxford-102-flowers'
        self.model = MobileNetV2(num_classes=102).to(device)
        self.train_dataset = ClassificationDataset(self.datadir_path, self.train_data_dir, 320, 320, aug=data_transforms('train'),load_img=load_iamge)
        self.val_dataset = ClassificationDataset(self.datadir_path, self.val_data_dir, 320, 320, aug=data_transforms('val'),load_img=load_iamge)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)

        self.cirterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)




    def train(self):
        train_data_size = len(self.train_dataset)
        val_data_size = len(self.val_dataset)
        print("训练数据集的长度为：{}".format(train_data_size))
        print("测试数据集的长度为：{}".format(val_data_size))
        print('Starting training for %g epochs...')
        epoch = 10

        val_num = len(self.val_dataset)
        train_steps = len(self.train_loader)

        for i in range(epoch):
            self.model.train()
            running_loss = 0.0
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                self.optimizer.zero_grad()
                logits = self.model(images.to(device))
                loss = self.cirterion(logits, labels.to(device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1,
                                                                         epoch,
                                                                         loss)

            # validate
            self.model.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = self.model(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(i + 1,
                                                               epoch)

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (i + 1, running_loss / train_steps, val_accurate))


if __name__ == '__main__':
    model = Classification()
    model.train()

