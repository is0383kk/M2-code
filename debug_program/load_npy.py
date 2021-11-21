import os
import numpy as np
from PIL import Image
#499物体,81カテゴリ
os.makedirs("./obj_data", exist_ok=True)
dataset = np.load('./obj_vision.npy')
label = np.loadtxt("./obj_label.txt")
print(f"label : {len(label)}, {label.shape}")
for i in range(len(label)):
    #os.makedirs("./obj_data/"+str(int(label[i])), exist_ok=True)
    data = dataset[i]
    scale = 255.0 / np.max(data)
    image = Image.fromarray(np.uint8(data*scale))
    image = image.resize((28, 28))
    image.save('./obj_data/'+str(int(label[i]))+'/obj'+str(i)+'.png')
    #print(int(label[i]))