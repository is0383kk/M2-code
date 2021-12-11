import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 

root = "../obj_data/train" # データセット読み込み先パス
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform = None, train = True):
        train_path = "train5"
        #path = "/home/is0383kk/workspace/mnist_png/training"
        classes = os.listdir(root+"/"+str(train_path))
        print(f"classes : {classes}")
        self.transform = transform # 前処理クラス
        # 画像とラベルの一覧を保持するリスト
        self.images = []; self.labels = []
        data_path = []
        image_path = []
        label_list = []
        # trainとval用で分ける
        if train == True:
            for i in range(len(classes)):
                data_path.append(os.path.join(root, str(train_path), classes[i]))
        else:
            for i in range(len(classes)):
                data_path.append(os.path.join(root, str(train_path), classes[i]))
        
        # 数字ごとの画像データ一覧を取得
        for i in range(len(classes)):
            image_path.append(os.listdir(data_path[i]))

        # 画像データにラベル付け
        for i in range(len(classes)):
            #print(f"[int(classes[i])] : {[int(classes[i])]}")
            #print(f"len(image_path[classes[i]]) : {len(image_path[i])}")
            label_list.append([int(classes[i])] * len(image_path[i]))
        
        for i in range(len(classes)):
            for image, label in zip(image_path[i], label_list[i]):
                self.images.append(os.path.join(data_path[i], image))
                self.labels.append(label)

    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得
        image = self.images[index]
        label = self.labels[index]
        # 画像ファイルパスから画像を読み込む
        with open(image, "rb") as f:
            image = Image.open(f)
            #image = image.resize((28, 28), Image.LANCZOS)
            #image = image.convert("L")
            image = image.convert("RGB")
        # 前処理
        if self.transform is not None:
            image = self.transform(image)
        
        # 画像とラベルのペアを返す
        return image, label

    def __len__(self):
        # データ数を指定
        return len(self.images)

class GenDataset(torch.utils.data.Dataset):
    classes = ["0","1","2","3","4","5","6","7","8","9"]
    
    def __init__(self, root, transform = None, train = True):
        self.transform = transform # 前処理クラス
        # 画像ラベルの一覧を保持するリスト
        self.images = []
        self.labels = []
        # データセット読み込み先パス
        root = "/home/is0383kk/workspace/mnist_png"
        # trainとval用で分ける
        if train == True:
            mnist_0_path = os.path.join(root, "gen", "0")
            mnist_1_path = os.path.join(root, "gen", "1")
            mnist_2_path = os.path.join(root, "gen", "2")
            mnist_3_path = os.path.join(root, "gen", "3")
            mnist_4_path = os.path.join(root, "gen", "4")
            mnist_5_path = os.path.join(root, "gen", "5")
            mnist_6_path = os.path.join(root, "gen", "6")
            mnist_7_path = os.path.join(root, "gen", "7")
            mnist_8_path = os.path.join(root, "gen", "8")
            mnist_9_path = os.path.join(root, "gen", "9")
        else:
            mnist_0_path = os.path.join(root, "testing", "0")
        
        # 数字ごとの画像データ一覧を取得
        mnist_image_0 = os.listdir(mnist_0_path)
        mnist_image_1 = os.listdir(mnist_1_path)
        mnist_image_2 = os.listdir(mnist_2_path)
        mnist_image_3 = os.listdir(mnist_3_path)
        mnist_image_4 = os.listdir(mnist_4_path)
        mnist_image_5 = os.listdir(mnist_5_path)
        mnist_image_6 = os.listdir(mnist_6_path)
        mnist_image_7 = os.listdir(mnist_7_path)
        mnist_image_8 = os.listdir(mnist_8_path)
        mnist_image_9 = os.listdir(mnist_9_path)
        # 画像データにラベル付け
        mnist_label_0 = [0] * len(mnist_image_0)
        mnist_label_1 = [1] * len(mnist_image_1)
        mnist_label_2 = [2] * len(mnist_image_2)
        mnist_label_3 = [3] * len(mnist_image_3)
        mnist_label_4 = [4] * len(mnist_image_4)
        mnist_label_5 = [5] * len(mnist_image_5)
        mnist_label_6 = [6] * len(mnist_image_6)
        mnist_label_7 = [7] * len(mnist_image_7)
        mnist_label_8 = [8] * len(mnist_image_8)
        mnist_label_9 = [9] * len(mnist_image_9)

        for image, label in zip(mnist_image_0, mnist_label_0):
            self.images.append(os.path.join(mnist_0_path, image))
            self.labels.append(label)
        # 1つのリスト構造にする 
        for image, label in zip(mnist_image_1, mnist_label_1):
            self.images.append(os.path.join(mnist_1_path, image))
            self.labels.append(label)
        # 1つのリスト構造にする 
        for image, label in zip(mnist_image_2, mnist_label_2):
            self.images.append(os.path.join(mnist_2_path, image))
            self.labels.append(label)
            # 1つのリスト構造にする 
        for image, label in zip(mnist_image_3, mnist_label_3):
            self.images.append(os.path.join(mnist_3_path, image))
            self.labels.append(label)
            # 1つのリスト構造にする 
        for image, label in zip(mnist_image_4, mnist_label_4):
            self.images.append(os.path.join(mnist_4_path, image))
            self.labels.append(label)
            # 1つのリスト構造にする 
        for image, label in zip(mnist_image_5, mnist_label_5):
            self.images.append(os.path.join(mnist_5_path, image))
            self.labels.append(label)
        for image, label in zip(mnist_image_6, mnist_label_6):
            self.images.append(os.path.join(mnist_6_path, image))
            self.labels.append(label)
        for image, label in zip(mnist_image_7, mnist_label_7):
            self.images.append(os.path.join(mnist_7_path, image))
            self.labels.append(label)
        for image, label in zip(mnist_image_8, mnist_label_8):
            self.images.append(os.path.join(mnist_8_path, image))
            self.labels.append(label)
        for image, label in zip(mnist_image_9, mnist_label_9):
            self.images.append(os.path.join(mnist_9_path, image))
            self.labels.append(label)

    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得
        image = self.images[index]
        label = self.labels[index]
        # 画像ファイルパスから画像を読み込む
        with open(image, "rb") as f:
            image = Image.open(f)
            #image = image.resize((28, 28), Image.LANCZOS)
            image = image.convert("L")
        # 前処理
        if self.transform is not None:
            image = self.transform(image)
        
        # 画像とラベルのペアを返す
        return image, label

    def __len__(self):
        # データ数を指定
        return len(self.images)