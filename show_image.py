import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 
from custom_data import CustomDataset, GenDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import matplotlib.animation as animation


root = "/home/is0383kk/workspace/mnist_png/mnist_png"
root = "/home/is0383kk/workspace/fruits-360_dataset/fruits-360"
root = "/home/is0383kk/workspace/dataset_test"
#root = "../obj_data/train" # データセット読み込み先パス
#前処理
data_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
#前処理なし
to_tensor = transforms.Compose([
    transforms.ToTensor(),
])
pad1 = transforms.Compose([
    transforms.Pad(padding=(0, -5, -10, 0), fill=0, padding_mode='constant'),transforms.Resize((28, 28)),# 左・上・右・下
    transforms.Pad(padding=(0, 5, 10, 0), fill=0, padding_mode='constant'),transforms.Resize((28, 28)),
    transforms.CenterCrop(77),
    transforms.ToTensor(),
])
pad2 = transforms.Compose([
    transforms.Pad(padding=(-15, 0, 0, 0), fill=0, padding_mode='constant'),transforms.Resize((28, 28)),# 左・上・右・下
    transforms.Pad(padding=(15, 0, 0, 0), fill=0, padding_mode='constant'),transforms.Resize((28, 28)),
    #transforms.CenterCrop(28),
    transforms.ToTensor(),
])
trans = transforms.Compose([
    transforms.ToTensor(),
])
trans_ang1 = transforms.Compose([
    #transforms.Resize((120, 120)),
    transforms.Resize((77, 77)),
    transforms.RandomRotation(degrees=(0,0),fill=256),
    
    transforms.ToTensor(),
])
trans_ang2 = transforms.Compose([
    transforms.RandomRotation(degrees=(25,25)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class ToNDarray(object):
    def __init__(self):
        pass

    def __call__(self, x):
        x_shape = x.shape    #x=(C,H,W)
        x = x.detach().clone().cpu()   #x=(C,H,W)
        x = x.numpy()   #x=(C,H,W)
        if x_shape[0] == 1:       #C=1の時
            x = x[0]    #x=(H,W)にする
        else:
            x = x.transpose(1,2,0)  #x=(H,W,C)にする
        return x

# データのプロット
custom_dataset = CustomDataset(root, trans_ang1, train=True)
#custom_dataset = CustomDataset(root, data_transforms, train=True)
#print(f"データセット数 :{len(custom_dataset)}")
batch_size = 10
custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False)
#print("batch_size",batch_size)
"""
# データセット分割調整
trainval_dataset = datasets.MNIST('./../../data', train=True, transform=pad1, download=False)
n_samples = len(trainval_dataset) 
#train_size = int(n_samples * 0.1) # 6000枚
train_size = int(n_samples * 0.13) # 9000枚
#train_size = int(n_samples * 0.25) # 15000枚
#train_size = int(n_samples * 0.01) # 600枚
#print(f"Number of training datasets :{train_size}")
subset1_indices = list(range(0,train_size)); subset2_indices = list(range(train_size,n_samples)) 
train_dataset = Subset(trainval_dataset, subset1_indices); val_dataset = Subset(trainval_dataset, subset2_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
"""
"""
# CIFAR10用
print("Dataset : CIFAR10")
angle_a = 0 # 回転角度
angle_b = -45 # 回転角度
trans_ang1 = transforms.Compose([transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.Resize((28, 28)), transforms.ToTensor()]) # -angle度回転設定
trans_ang2 = transforms.Compose([transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.Resize((28, 28)), transforms.ToTensor()]) # angle度回転設定
trainval_dataset1 = datasets.CIFAR10(root='./../../data', train=True, download=False, transform=trans_ang1)
trainval_dataset2 = datasets.CIFAR10(root='./../../data', train=True, download=False, transform=trans_ang2)
n_samples = len(trainval_dataset1)
print(f"n_samples : {n_samples}")
D = int(n_samples * (1/5)) # データ総数
#D = int(n_samples) # データ総数
subset1_indices1 = list(range(0, D)); subset2_indices1 = list(range(D, n_samples)) 
subset1_indices2 = list(range(0, D)); subset2_indices2 = list(range(D, n_samples)) 
train_dataset1 = Subset(trainval_dataset1, subset1_indices1); val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
train_dataset2 = Subset(trainval_dataset2, subset1_indices1); val_dataset2 = Subset(trainval_dataset2, subset2_indices2)
train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=16, shuffle=False) # train_loader for agent A
train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=16, shuffle=False) # train_loader for agent B
all_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=D, shuffle=False) # データセット総数分のローダ
all_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=D, shuffle=False) # データセット総数分のローダ
"""

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,1,0)), interpolation="nearest")
    plt.show()

for i, (images, labels) in enumerate(custom_loader):
    print(f"i :{i}")
    #print(f"images :{images}, {images.size()}")
    trans = ToNDarray()
    print(f"labels :{labels}")
    #print(images)
    #print(labels.numpy())
    #im = trans(images[0])
    #print(f"im: {im}")
    #transform = transforms.FiveCrop(24)
    #imgs = transform(im)
    images = torchvision.utils.make_grid(images, padding=1)
    plt.imshow(np.transpose(images, (1,2,0)), interpolation="nearest")
    plt.savefig('fruit10.png')
    
    plt.show()
    plt.close()
    #show(torchvision.utils.make_grid(images, padding=1))
    #plt.axis("off")

    break

