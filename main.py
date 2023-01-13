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
    parser.add_argument('--model', type=list, default=['mobilenetv1','mobilenetv2','mobilenetv3','resnet18',
                                                       'resnet34','resnet50','resnet101','resnet18','resnet152',
                                                       'resnext50_32x4d','resnext101_32x8d','wide_resnet101_2','wide_resnet50_2',
                                                       'shufflenetv1','shufflenetv2','shufflenetv2p','squeezenetv1.0',
                                                       'squeezenetv1.1','densenet121','densenet169','densenet201','densenet265',
                                                       'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4',
                                                       'efficientnetb5','efficientnetb6','efficientnetb7',
                                                       'efficientnets','efficientnetm','efficientnetl','ghostnet',
                                                       'mnasnet','peleenet'], help='mobilenetv1,mobilenetv2,mobilenetv3,resnet18,resnet34,resnet50,resnet101,resnet18,\
                                                        resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet101_2,wide_resnet50_2,shufflenetv1,shufflenetv2,shufflenetv2p,\
                                                        squeezenetv1.0,squeezenetv1.1,densenet121,densenet169,densenet201,densenet265,efficientnetb0,efficientnetb1,efficientnetb2,\
                                                        efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7,efficientnets,efficientnetm,efficientnetl,\
                                                        ghostnet,mnasnet,peleenet')
    parser.add_argument('--classes', type=int, default=102, help='number of categories')
    parser.add_argument('--train_path', type=str, default='D:/xunleidownload/Oxford_102_Flowers/data/oxford-102-flowers', help='train dataset path')
    parser.add_argument('--test_path', type=str, default='D:/xunleidownload/Oxford_102_Flowers/data/oxford-102-flowers', help='test dataset path')
    parser.add_argument('--epoch',type=str,default=1,help='epochs')
    parser.add_argument('--size', type=int, default=320, help='iamge width')
    parser.add_argument('--batch_size', type=int, default= 32, help='batch size')
    parser.add_argument('--nw', type=int, default=8, help='num workers')
    parser.add_argument('--dp', action="store_true", default=False, help='DataParallel')
    parser.add_argument('--lr', type=int, default= 0.001, help='learning rate')
    parser.add_argument('--aug', type=str, default="baseline", help='choose optimizer[adam,sgd,sgdmomentum,adagrad,radam,rmsprop,adamw,asgd,adadelta]')
    parser.add_argument('--optmi', type=str, default="adam", help='choose aug[baseline,autoaugment,randaugment,trivialaugment,randomerasing]')
    parser.add_argument('--train', action="store_true", default=False, help='train or test')

    opt = parser.parse_args()

    # print(len(opt.model))
    for i in range(len(opt.model)):
        print("******************************{} model start training***************************************".format(opt.model[i]))
        if opt.train:
            model = Classification_Train(opt.model[i],opt.aug,opt.lr,opt.optmi,opt.classes,opt.epoch,opt.batch_size,opt.nw,opt.dp,opt.train_path,opt.size)
            model.train()
            print("******************************{} model end of training***************************************".format(opt.model[i]))
            print("------------------------------{} model start testing---------------------------------------".format(opt.model[i]))
            model = Classification_Test(opt.model[i],opt.aug,opt.classes,opt.test_path,opt.size)
            model.test()
        print("------------------------------{} model end of testing---------------------------------------".format(opt.model[i]))