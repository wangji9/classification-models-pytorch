from PIL import Image
# from torchvision import transforms
from torchvision.transforms import autoaugment, transforms

import torch
def load_iamge(iamge_path):
    img = Image.open(iamge_path).convert('RGB')
    return img

# def data_transforms(x,width,height):
#     """
#
#     :param x:
#     :param width:
#     :param height:
#     :return:
#     """
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize((width,height)),
#             transforms.CenterCrop((width,height)),
#             # 转换成tensor向量
#             transforms.ToTensor(),
#             # 对图像进行归一化操作
#             # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize((width,height)),
#             transforms.CenterCrop((width,height)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#     return data_transforms[x]


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
def getaug(aug,size):

    global train_transform, test_transform
    if aug == "baseline":
        # 训练
        train_transform = transforms.Compose([
            # 这里的scale指的是面积，ratio是宽高比
            # 具体实现每次先随机确定scale和ratio，可以生成w和h，然后随机确定裁剪位置进行crop
            # 最后是resize到target size
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
         ])
        # 测试
        test_transform = transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
         ])


    if aug == "autoaugment":
        aa_policy = autoaugment.AutoAugmentPolicy('imagenet')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            # 这里policy属于torchvision.transforms.autoaugment.AutoAugmentPolicy，
            # 对于ImageNet就是 AutoAugmentPolicy.IMAGENET
            # 此时aa_policy = autoaugment.AutoAugmentPolicy('imagenet')

            autoaugment.AutoAugment(policy=aa_policy),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ])
        # 测试
        test_transform = transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if aug == "randaugment":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            autoaugment.RandAugment(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ])
        # 测试
        test_transform = transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if aug == "trivialaugment":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            autoaugment.TrivialAugmentWide(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ])
        # 测试
        test_transform = transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    if aug == "randomerasing":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
            # scale是指相对于原图的擦除面积范围
            # ratio是指擦除区域的宽高比
            # value是指擦除区域的值，如果是int，也可以是tuple（RGB3个通道值），或者是str，需为'random'，表示随机生成
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ])
        # 测试
        test_transform = transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return train_transform, test_transform


if __name__ == '__main__':
    train_transform,test_transform = getaug("randomerasing",224)
