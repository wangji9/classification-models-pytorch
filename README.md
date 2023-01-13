# Classification-models
## dataset:Oxford 102 Flower (102 Category Flower Dataset)  
Homepage:https://www.robots.ox.ac.uk/~vgg/data/flowers/102/


2023/1/11:The basic training and testing framework of image classification is created  
2023/1/12:add log;add argparse;add some models  
2023/1/13:add data augmentations;add optimizers;modified code structure  
# model  
    mobilenetv1  
    mobilenetv2  
    mobilenetv3
    resnets  
    shufflenents  
    squeezenets  
    densenets  
    efficientnetv1s  
    efficientnetv2s  
    ghostnet  
    mnasnet  
    peleenet
    ...
# data augmentations    
    baseline,autoaugment,randaugment,trivialaugment,randomerasing  
# optimizers  
    adam,sgd,sgdmomentum,adagrad,radam,rmsprop,adamw,asgd,adadelta  
usage: main.py [-h] [--model MODEL] [--classes CLASSES] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--epoch EPOCH] [--size SIZE] [--batch_size BATCH_SIZE] [--nw NW] [--dp] [--lr LR] [--aug AUG] [--optmi OPTMI] [--train]  
                                                                                                                                                                                                                                  
optional arguments:                                                                                                                                                                                                                
      -h, --help            show this help message and exit                                                                                                                                                                             
      --model MODEL         mobilenetv1,mobilenetv2,mobilenetv3,resnet18,resnet34,resnet50,resnet101,resnet18, resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet101_2,wide_resnet50_2,shufflenetv1,shufflenetv2,shufflenetv2p,
                            squeezenetv1.0,squeezenetv1.1,densenet121,densenet169,densenet201,densenet265,efficientnetb0,efficientnetb1,efficientnetb2, efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7,efficientnets,efficientnetm,efficientnetl, ghostnet,mnasnet,peleenet  
      --classes CLASSES     number of categories  
      --train_path TRAIN_PATH
                            train dataset path  
      --test_path TEST_PATH
                            test dataset path  
      --epoch EPOCH         epochs  
      --size SIZE           iamge width  
      --batch_size BATCH_SIZE
                            batch size  
      --nw NW               num workers  
      --dp                  DataParallel  
      --lr LR               learning rate  
      --aug AUG             choose optimizer[adam,sgd,sgdmomentum,adagrad,radam,rmsprop,adamw,asgd,adadelta]  
      --optmi OPTMI         choose aug[baseline,autoaugment,randaugment,trivialaugment,randomerasing]  
      --train               train or test  

## train  
python main.py --train  
## test  
python main.py  
## reference  
https://github.com/megvii-model/ShuffleNet-Series
