import os
import torch

from augmentation import load_iamge, data_transforms
from get_models import getModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Classification_Test():
    def __init__(self,models,classes,test_path,width,height) -> None:
        """
        :param models:
        :param classes:
        :param test_path:
        :param width:
        :param height:
        """
        self.datadir_path = test_path
        self.test_path = self.datadir_path + '/'+'test.txt'
        self.modelname = models
        self.model = getModel(models,classes).to(device)
        self.aug = data_transforms('val',width,height)

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

            with open('log/test_log.txt', "a+", encoding="utf-8") as f:
                f.write("index: {:3},class: {:3},label: {:3},prob: {:.3}\n".format(i,predict_cla, self.labels[i], predict[predict_cla].numpy()))
                f.close()
        with open('log/test_log.txt', "a+", encoding="utf-8") as f:
            f.write("acc:{}\n".format(a / len(self.images)))
            f.close()
        print("acc:{}".format(a / len(self.images)))

if __name__ == '__main__':
    model = Classification_Test()
    model.test()