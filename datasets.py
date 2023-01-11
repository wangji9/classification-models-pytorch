from PIL import Image
import os
from torch.utils.data import DataLoader,Dataset

from torchvision import transforms

def load_iamge(iamge_path):
    img = Image.open(iamge_path).convert('RGB')
    return img

def data_transforms(x):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(320),
            # 转换成tensor向量
            transforms.ToTensor(),
            # 对图像进行归一化操作
            # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms[x]


class ClassificationDataset(Dataset):
    def __init__(self,datadir_path,path,img_width,img_height,load_img,aug):
        self.datadir_path = datadir_path
        self.path = path
        self.img_width = img_width
        self.img_height = img_height
        self.aug = aug
        self.load_img = load_img

        # self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
        fp = open(self.path, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(self.datadir_path+'/'+information[0])
            labels.append(int(information[1]))
        self.images = images
        self.labels = labels


    def __getitem__(self, index):
        iamge_path = self.images[index]
        label = self.labels[index]
        image = self.load_img(iamge_path)
        image = image.resize((self.img_width,self.img_height))

        if self.aug:
            image = self.aug(image)
        else:
            pass
        return image,label


    def __len__(self):
        return len(self.images)



if __name__ == '__main__':
    train_data_dir = 'Oxford_102_Flowers/data/oxford-102-flowers/train.txt'
    val_data_dir = 'Oxford_102_Flowers/data/oxford-102-flowers/valid.txt'
    datadir_path = 'Oxford_102_Flowers/data/oxford-102-flowers'
    train_dataset = ClassificationDataset(datadir_path,train_data_dir,320,320,aug=data_transforms('train'),load_img=load_iamge)
    val_dataset = ClassificationDataset(datadir_path,val_data_dir,320,320,aug=data_transforms('val'),load_img=load_iamge)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(train_dataset.__getitem__(0))
    print(val_dataset.__getitem__(0))
    print(train_dataset.__len__())
    print(val_dataset.__len__())