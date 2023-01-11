import torch
import sys
import warnings

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from PIL import Image
from datasets import ClassificationDataset, load_iamge, data_transforms
from plot import plot_loss, plot_acc
from tqdm import tqdm

warnings.filterwarnings("ignore")

from models.mobilenetv1 import MobileNetV1
from models.mobilenetv3 import MobileNetV3
from models.mobilenetv2 import MobileNetV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classification_Train():
    def __init__(self):

        self.train_data_dir = 'Oxford_102_Flowers/data/oxford-102-flowers/train.txt'
        self.val_data_dir = 'Oxford_102_Flowers/data/oxford-102-flowers/valid.txt'
        self.datadir_path = 'Oxford_102_Flowers/data/oxford-102-flowers'

        # self.model = torch.nn.DataParallel(MobileNetV2(num_classes=102)).to(device)
        self.model = MobileNetV2(num_classes=102).to(device)

        self.train_dataset = ClassificationDataset(self.datadir_path, self.train_data_dir, 320, 320,
                                                   aug=data_transforms('train'), load_img=load_iamge)
        self.val_dataset = ClassificationDataset(self.datadir_path, self.val_data_dir, 320, 320,
                                                 aug=data_transforms('val'), load_img=load_iamge)

        self.train_loader = DataLoader(self.train_dataset, batch_size=160, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.val_dataset, batch_size=160, shuffle=False, num_workers=8)

        self.cirterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        train_data_size = len(self.train_dataset)
        val_data_size = len(self.val_dataset)
        print("训练数据集的长度为：{}".format(train_data_size))
        print("测试数据集的长度为：{}".format(val_data_size))

        epoch = 20
        best_acc = 0.0
        save_path = 'model.pth'
        print('Starting training for {} epochs...'.format(epoch))

        val_num = len(self.val_dataset)
        train_num = len(self.train_dataset)
        train_steps = len(self.train_loader)
        val_steps = len(self.val_loader)
        train_loss_list = []
        val_loss_list = []
        train_accurate_list = []
        val_accurate_list = []
        epoch_list = []

        for i in range(epoch):
            self.model.train()
            train_acc = 0.0
            running_loss = 0.0
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                self.optimizer.zero_grad()
                logits = self.model(images.to(device))
                loss = self.cirterion(logits, labels.to(device))
                loss.backward()
                predict_y = torch.max(logits, dim=1)[1]
                train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
                self.optimizer.step()
                running_loss += loss.item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1, epoch, loss)

            # validate
            self.model.eval()
            val_acc = 0.0
            valing_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = self.model(val_images.to(device))
                    loss = self.cirterion(outputs, val_labels.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    valing_loss += loss.item()
                    val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}]".format(i + 1, epoch, loss)

            train_accurate = train_acc / train_num
            train_accurate_list.append(train_accurate)
            val_accurate = val_acc / val_num
            val_accurate_list.append(val_accurate)
            train_loss = running_loss / train_steps
            val_loss = valing_loss / val_steps
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            epoch_list.append(i)

            plot_loss(epoch_list, train_loss_list, val_loss_list)
            plot_acc(epoch_list, train_accurate_list, val_accurate_list)
            print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' % (
            i + 1, train_loss, train_accurate, val_loss, val_accurate))
            print(train_accurate_list, val_accurate_list, train_loss_list, val_loss_list)

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(self.model.state_dict(), save_path)
                print("best model is save")

        print('Finished Training')


if __name__ == '__main__':
    model = Classification_Train()
    model.train()


