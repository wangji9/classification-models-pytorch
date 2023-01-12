import torch
import sys

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from datasets import ClassificationDataset
from augmentation import load_iamge, data_transforms
from plot import plot_loss, plot_acc
from tqdm import tqdm
from get_models import getModel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classification_Train():

    def __init__(self,model,classes,epoch,batch_size,nw,dp,datadir_path,width,height):
        """
        :param models:
        :param classes:
        :param epoch:
        :param datadir_path:
        :param width:
        :param height:
        """
        self.datadir_path = datadir_path
        self.train_data_dir = self.datadir_path + '/'+'train.txt'
        self.val_data_dir = self.datadir_path + '/'+'valid.txt'
        self.modelname = model
        self.epoch = epoch
        self.batch_size = batch_size
        self.nw = nw
        self.dp = dp
        if self.dp:
            self.model = torch.nn.DataParallel(getModel(model,classes)).to(device)
        else:
            self.model = getModel(model,classes).to(device)

        self.train_dataset = ClassificationDataset(self.datadir_path, self.train_data_dir, 320, 320,
                                                   aug=data_transforms('train',width,height), load_img=load_iamge)
        self.val_dataset = ClassificationDataset(self.datadir_path, self.val_data_dir, 320, 320,
                                                 aug=data_transforms('val',width,height), load_img=load_iamge)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.nw)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.nw)

        self.cirterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        train_data_size = len(self.train_dataset)
        val_data_size = len(self.val_dataset)
        print("训练数据集的长度为：{}".format(train_data_size))
        print("测试数据集的长度为：{}".format(val_data_size))

        best_acc = 0.0
        save_path = 'checkpoint/model_{}.pth'.format(self.modelname)
        print('Starting training for {} epochs...'.format(self.epoch))

        val_num = len(self.val_dataset)
        train_num = len(self.train_dataset)
        train_steps = len(self.train_loader)
        val_steps = len(self.val_loader)

        train_loss_list = []
        val_loss_list = []
        train_accurate_list = []
        val_accurate_list = []
        epoch_list = []

        for i in range(self.epoch):
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
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1, self.epoch, loss)

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
                    val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}]".format(i + 1, self.epoch, loss)

            train_accurate = train_acc / train_num
            train_accurate_list.append(train_accurate)
            val_accurate = val_acc / val_num
            val_accurate_list.append(val_accurate)
            train_loss = running_loss / train_steps
            val_loss = valing_loss / val_steps
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            epoch_list.append(i)

            plot_loss(epoch_list, train_loss_list, val_loss_list,self.modelname)
            plot_acc(epoch_list, train_accurate_list, val_accurate_list,self.modelname)
            print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' % (i + 1, train_loss, train_accurate, val_loss, val_accurate))
            # print(train_accurate_list, val_accurate_list, train_loss_list, val_loss_list)

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(self.model.state_dict(), save_path)
                print("best model is save")

            with open('log/train_log.txt',"a+", encoding="utf-8") as f:
                f.write('index: {:3},epoch:{:3} train_loss:{:3f} val_loss:{:3f} train_acc:{:3f} val_acc:{:3f}\n'.format(i,epoch_list[i],train_loss_list[i],val_loss_list[i],train_accurate_list[i],val_accurate_list[i]))
                f.close()


        print('Finished Training')


if __name__ == '__main__':
    model = Classification_Train()
    model.train()


