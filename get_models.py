from models.mobilenetv1 import MobileNetV1
from models.mobilenetv3 import MobileNetV3
from models.mobilenetv2 import MobileNetV2

def getModel(model,classes):

    if model == 'mobilenetv1':
        model = MobileNetV1(
            num_classes=classes
        )
    if model == 'mobilenetv2':
        model = MobileNetV2(
            num_classes=classes
        )
    if model == 'mobilenetv3':
        model = MobileNetV3(
            num_classes=classes
        )
    return model