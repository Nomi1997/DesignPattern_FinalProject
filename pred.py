import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

def Predict_loader(image_dir, resize):
    img_list = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for image_name in tqdm(os.listdir(image_dir)):
        path = os.path.join(image_dir, image_name)
        _, fType = os.path.splitext(path)
        if fType == ".jpg":
            img = read(path)
            img = cv2.resize(img, (resize, resize))
            img_list.append(np.array(img)/255.)

    return np.array(img_list)

def drawPredict(image, predict, index):
    plt.imshow(image[index])
    plt.title(predict[index])
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    image_dir = "./HAM10000_images_test/"
    testImage = Predict_loader(image_dir, 224)
    path = "./test.h5"
    model = tf.keras.models.load_model(path)

    predict = model.predict(testImage)

    drawPredict(testImage, predict, 0)
