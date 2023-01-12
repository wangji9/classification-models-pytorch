from PIL import Image
from torchvision import transforms

def load_iamge(iamge_path):
    img = Image.open(iamge_path).convert('RGB')
    return img

def data_transforms(x,width,height):
    """

    :param x:
    :param width:
    :param height:
    :return:
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((width,height)),
            transforms.CenterCrop((width,height)),
            # 转换成tensor向量
            transforms.ToTensor(),
            # 对图像进行归一化操作
            # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((width,height)),
            transforms.CenterCrop((width,height)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[x]
