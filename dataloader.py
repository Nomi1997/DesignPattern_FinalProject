import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class Dataloader():
    def Dataset_loader(image_dir, label_info, resize):
        img_list = []
        lab_list = []
        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
        for image_name in tqdm(os.listdir(image_dir)):
            path = os.path.join(image_dir, image_name)
            _, fType = os.path.splitext(path)
            if fType == ".jpg":
                img = read(path)
                img = cv2.resize(img, (resize, resize))
                lab = label_info[label_info['image_id'].isin([image_name[:-4]])]['dx'].values[0]
                
                img_list.append(np.array(img)/255.)
                lab_list.append(lab)

        return np.array(img_list), np.array(lab_list)

    def split_Train_Validation(img_list, lab_list, radio = 0.8):
        num = int(radio * len(img_list))
        
        (train_inputs, val_inputs) = (img_list[:num], img_list[num:])
        (train_label, val_label) = (lab_list[:num], lab_list[num:])

        return train_inputs, train_label, val_inputs, val_label

if __name__ == '__main__':
    image_dir = "./HAM10000_images/"
    metaDataCSV = "./HAM10000_metadata.csv"
    metaDataCSVInfo = pd.read_csv(metaDataCSV, encoding='utf-8')

    uni = sorted(metaDataCSVInfo['dx'].unique())
    for i in range(len(uni)):
        metaDataCSVInfo['dx'].replace(uni[i], i , inplace=True)

    img_list, lab_list = Dataloader.Dataset_loader(image_dir, metaDataCSVInfo[['image_id', 'dx']], 224)
    # train_inputs, train_label, val_inputs, val_label = split_Train_Validation(img_list, lab_list, 0.8)
    train_inputs, val_inputs, train_label, val_label = train_test_split(img_list, lab_list, test_size = 0.2, random_state = 42)
