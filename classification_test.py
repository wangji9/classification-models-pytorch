import os
import torch

from utils.get_aug import load_iamge
from utils.get_models import getModel
from utils.get_aug import getaug
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Classification_Test():
    def __init__(self,model,aug,classes,test_path,size) -> None:
        """
        :param models:
        :param classes:
        :param test_path:
        :param width:
        :param height:
        """
        self.size = size

        train_transform, test_transform = getaug(aug, self.size)
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.datadir_path = test_path
        self.test_path = self.datadir_path + '/'+'test.txt'
        self.modelname = model
        self.model = getModel(model,classes).to(device)


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
        with open('log/{}_test_log.txt'.format(self.modelname), "w", encoding="utf-8") as f:
            f.write('')
            f.close()
        predict_cla_list = []
        a = 0
        for i, iamge in enumerate(self.images):
            iamge = load_iamge(iamge)
            iamge = self.test_transform(iamge)
            img = torch.unsqueeze(iamge, dim=0)
            weights_path = 'checkpoint/model_{}.pth'.format(self.modelname)
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

            with open('log/{}_test_log.txt'.format(self.modelname), "a+", encoding="utf-8") as f:
                f.write("index: {:3},class: {:3},label: {:3},prob: {:.3}\n".format(i,predict_cla, self.labels[i], predict[predict_cla].numpy()))
                f.close()
        with open('log/{}_test_log.txt'.format(self.modelname), "a+", encoding="utf-8") as f:
            f.write("acc:{}\n".format(a / len(self.images)))
            f.close()
        print("acc:{}".format(a / len(self.images)))

if __name__ == '__main__':
    model = Classification_Test()
    model.test()