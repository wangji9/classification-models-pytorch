# Classification-models
## dataset:Oxford 102 Flower (102 Category Flower Dataset)  
Homepage:https://www.robots.ox.ac.uk/~vgg/data/flowers/102/


2023/1/11:创建了图像分类的基本训练、测试框架  
2023/1/12：add log;add argparse；add some models  
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

usage: main.py [-h] [--models MODELS] [--classes CLASSES] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--epoch EPOCH] [--width WIDTH] [--height HEIGHT] [--train]  
optional arguments:
>   -h, --help            show this help message and exit  
    --models MODELS       choose models  
    --classes CLASSES     number of categories  
    --train_path TRAIN_PATH
                        train dataset path  
    --test_path TEST_PATH
                        test dataset path  
    --epoch EPOCH         epochs  
    --width WIDTH         iamge width  
    --height HEIGHT       iamge height  
    --train               train or test
## train  
python main.py --train  
## test  
python main.py  

https://github.com/megvii-model/ShuffleNet-Series
