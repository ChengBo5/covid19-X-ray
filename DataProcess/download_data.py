import os
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image




def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_Non,txt_CP, transform=None):
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_Non, txt_CP]
        self.classes = ['CT_COVID', 'CT_Non', 'CT_CP']
        self.num_cls = len(self.classes)
        self.img_list = []

        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5), #依据概率p对PIL图片进行水平翻转
    transforms.RandomVerticalFlip(0.5),   #依据概率p对PIL图片进行垂直翻转
    torchvision.transforms.RandomAffine(degrees=(-30,30), translate=(0.1,0.1), scale=(0.9,1.1)),#仿射变换
    transforms.ToTensor(),
    normalize
])

test_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])


if __name__ == '__main__':
    print("data process")  

    trainset = CovidCTDataset(root_dir='./covid19_dataset',
                              txt_COVID='./covid19_dataset/COVID.txt',
                              txt_Non='./covid19_dataset/Normal.txt',
                              txt_CP='./covid19_dataset/CP.txt',
                              transform=train_transformer)    

    # testset = CovidCTDataset(root_dir='test_image',
    #                          txt_COVID='test_image/test_COVID.txt',
    #                          txt_Non='test_image/test_Normal.txt',
    #                          txt_CP='test_image/test_CP.txt',
    #                          transform=test_transformer)

    train_loader = DataLoader(trainset, batch_size=20, drop_last=False, shuffle=True)



