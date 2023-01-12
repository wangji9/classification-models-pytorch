from models.mobilenetv1 import MobileNetV1
from models.mobilenetv3 import MobileNetV3
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18,resnet34,resnet50,\
    resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,\
    wide_resnet101_2,wide_resnet50_2
from models.shufflenetv1 import ShuffleNetV1
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenetv2p import ShuffleNetV2p
from models.densenet import DenseNet121,DenseNet169,DenseNet201,DenseNet265
from models.squeezenet import SqueezeNet
from models.efficientnetv1 import efficientnet_b0,efficientnet_b1,efficientnet_b2,\
    efficientnet_b3,efficientnet_b4,efficientnet_b5,efficientnet_b6,efficientnet_b7
from models.efficientnetv2 import efficientnetv2_l,efficientnetv2_m,efficientnetv2_s
from models.ghostnet import ghost_net
from models.mnasnet import MnasNet
from models.peleenet import PeleeNet
def getModel(model,classes):
    #mobilenet
    if model == 'mobilenetv1':
        model = MobileNetV1(num_classes=classes)
    if model == 'mobilenetv2':
        model = MobileNetV2(num_classes=classes)
    if model == 'mobilenetv3':
        model = MobileNetV3(num_classes=classes)
    #resnet
    if model == 'resnet18':
        model = resnet18(num_classes=classes,pretrained=False)
    if model == 'resnet34':
        model = resnet34(num_classes=classes,pretrained=False)
    if model == 'resnet50':
        model = resnet50(num_classes=classes,pretrained=False)
    if model == 'resnet101':
        model = resnet101(num_classes=classes,pretrained=False)
    if model == 'resnet152':
        model = resnet152(num_classes=classes,pretrained=False)
    if model == 'resnext50_32x4d':
        model = resnext50_32x4d(num_classes=classes,pretrained=False)
    if model == 'resnext101_32x8d':
        model = resnext101_32x8d(num_classes=classes,pretrained=False)
    if model == 'wide_resnet101_2':
        model = wide_resnet101_2(num_classes=classes,pretrained=False)
    if model == 'wide_resnet50_2':
        model = wide_resnet50_2(num_classes=classes,pretrained=False)
    #shufflenent
    if model == "shufflenetv1":
        model = ShuffleNetV1(n_class=classes,group=3)
    if model == "shufflenetv2":
        model = ShuffleNetV2(n_class=classes)
    if model == "shufflenetv2p":
        model = ShuffleNetV2p(n_class=classes)
    #squeezenet
    if model == "squeezenetv1.0":
        model = SqueezeNet("1_0",n_class=classes)
    if model == "squeezenetv1.1":
        model = SqueezeNet("1_1",n_class=classes)
    #densenet
    if model == "densenet121":
        model = DenseNet121(n_class=classes)
    if model == "densenet169":
        model = DenseNet169(n_class=classes)
    if model == "densenet201":
        model = DenseNet201(n_class=classes)
    if model == "densenet265":
        model = DenseNet265(n_class=classes)
    #efficientnetv1
    if model == "efficientnetb0":
        model = efficientnet_b0(num_classes=classes)
    if model == "efficientnetb1":
        model = efficientnet_b1(num_classes=classes)
    if model == "efficientnetb2":
        model = efficientnet_b2(num_classes=classes)
    if model == "efficientnetb3":
        model = efficientnet_b3(num_classes=classes)
    if model == "efficientnetb4":
        model = efficientnet_b4(num_classes=classes)
    if model == "efficientnetb5":
        model = efficientnet_b5(num_classes=classes)
    if model == "efficientnetb6":
        model = efficientnet_b6(num_classes=classes)
    if model == "efficientnetb7":
        model = efficientnet_b7(num_classes=classes)
    ##efficientnetv2
    if model == "efficientnets":
        model = efficientnetv2_s(num_classes=classes)
    if model == "efficientnetsm":
        model = efficientnetv2_m(num_classes=classes)
    if model == "efficientnetl":
        model = efficientnetv2_l(num_classes=classes)

    if model == "ghostnet":
        model = ghost_net(num_classes=classes)
    if model == "mnasnet":
        model = MnasNet(n_class=classes)
    if model == "peleenet":
        model = PeleeNet(num_classes=classes)
    return model