import argparse
from classification_train import Classification_Train
from classification_test import Classification_Test
import os

def del_file(path_data):
    """
    :param path_data:
    :return:
    """
    for i in os.listdir(path_data) :# 返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i#当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):#os.path.isfile判断是否为文件,如果是文件,就删除
            os.remove(file_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default="mobilenetv1", help='choose models')
    parser.add_argument('--classes', type=int, default=102, help='number of categories')
    parser.add_argument('--train_path', type=str, default='D:/xunleidownload/Oxford_102_Flowers/data/oxford-102-flowers', help='train dataset path')
    parser.add_argument('--test_path', type=str, default='D:/xunleidownload/Oxford_102_Flowers/data/oxford-102-flowers', help='test dataset path')
    parser.add_argument('--epoch',type=str,default=10,help='epochs')
    parser.add_argument('--width', type=int, default='320', help='iamge width')
    parser.add_argument('--height', type=int, default='320', help='iamge height')

    parser.add_argument('--train', action="store_true", default=False, help='train or test')

    opt = parser.parse_args()

    if opt.train:
        del_file('log')
        model = Classification_Train(opt.models,opt.classes,opt.epoch,opt.train_path,opt.width,opt.height)
        model.train()
    else:
        model = Classification_Test(opt.models,opt.classes,opt.test_path,opt.width,opt.height)
        model.test()