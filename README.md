# Classification-models
## Dataset:Oxford 102 Flower (102 Category Flower Dataset)  
Homepage:https://www.robots.ox.ac.uk/~vgg/data/flowers/102/


2023/1/11:The basic training and testing framework of image classification is created  
2023/1/12:add log;add argparse;add some models  
2023/1/13:add data augmentations;add optimizers;modified code structure  
# Models  
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
# Data augmentations    
    baseline,autoaugment,randaugment,trivialaugment,randomerasing  
# Optimizers  
    adam,sgd,sgdmomentum,adagrad,radam,rmsprop,adamw,asgd,adadelta  
usage: main.py [-h] [--model MODEL] [--classes CLASSES] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--epoch EPOCH] [--size SIZE] [--batch_size BATCH_SIZE] [--nw NW] [--dp] [--lr LR] [--aug AUG] [--optmi OPTMI] [--train]  
                                                                                                                                                                                                                                  
optional arguments:                                                                                                                                                                                                                
      -h, --help            show this help message and exit                                                                                                                                                                             
      --model MODEL         mobilenetv1,mobilenetv2,mobilenetv3,resnet18,resnet34,resnet50,resnet101,resnet18,resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet101_2,wide_resnet50_2,shufflenetv1,shufflenetv2,shufflenetv2p,squeezenetv1.0,squeezenetv1.1,densenet121,densenet169,densenet201,densenet265,efficientnetb0,efficientnetb1,efficientnetb2,efficientnetb3,efficientnetb4,efficientnetb5,efficientnetb6,efficientnetb7,efficientnets,efficientnetm,efficientnetl, ghostnet,mnasnet,peleenet  
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

## Train  
python main.py --train  
## Test  
python main.py  
## Efficientnetb0 Result  
![Accuracy_efficientnetb0](https://user-images.githubusercontent.com/66462413/212268847-251a5939-79f3-4c56-a71e-5929115984db.jpg)
![Loss_efficientnetb0](https://user-images.githubusercontent.com/66462413/212268882-35e8a6c7-a54a-49f5-a158-fd934f85a758.jpg)  
## Different data augmentations methods  
# Accuracy  
![Accuracy_efficientnetb0_adam_baseline](https://user-images.githubusercontent.com/66462413/212460726-9d803f35-89f3-4545-98ec-08215c521490.jpg)
![Accuracy_efficientnetb0_adam_randaugment](https://user-images.githubusercontent.com/66462413/212460728-db640385-200b-4f92-9e04-4c33b57e3c72.jpg)
![Accuracy_efficientnetb0_adam_randomerasing](https://user-images.githubusercontent.com/66462413/212460730-23c58189-e956-4b27-9265-4bab56dbca8c.jpg)
![Accuracy_efficientnetb0_adam_trivialaugment](https://user-images.githubusercontent.com/66462413/212460733-5ab9f5f3-1d6f-4685-86a4-ba869688ac40.jpg)
![Accuracy_efficientnetb0_adam_autoaugment](https://user-images.githubusercontent.com/66462413/212460735-568ba05e-f517-4be3-afa0-4a8f3797a006.jpg)
# Loss  
![Loss_efficientnetb0_adam_trivialaugment](https://user-images.githubusercontent.com/66462413/212460751-57c01eb6-6d11-4c84-a39e-0b3f83a5754b.jpg)
![Loss_efficientnetb0_adam_autoaugment](https://user-images.githubusercontent.com/66462413/212460754-ac6f6d73-d546-4bc2-9c27-bf1b0add4057.jpg)
![Loss_efficientnetb0_adam_baseline](https://user-images.githubusercontent.com/66462413/212460755-4b003811-0b1b-43c2-bce0-7f0a9076f6a4.jpg)
![Loss_efficientnetb0_adam_randaugment](https://user-images.githubusercontent.com/66462413/212460756-9b6201cd-9076-4b8e-afb4-10c12b13474c.jpg)
![Loss_efficientnetb0_adam_randomerasing](https://user-images.githubusercontent.com/66462413/212460758-377eaeb3-7724-4cd4-91e2-62e4c4e66a60.jpg)
# Test  
| efficientnetb0+Adam+epoch=100 |          |             |             |               |                |
| ----------------------------- | -------- | ----------- | ----------- | ------------- | -------------- |
| Data Augmentations            | baseline | autoaugment | randaugment | randomerasing | trivialaugment |
| Accuracy                      | 83.24    | 88.53       | 88.04       | 83.14         | 88.53          |
## Reference  
https://github.com/megvii-model/ShuffleNet-Series
