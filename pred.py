import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

class Predict():
    def Predict_loader(self, image_dir, resize):
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

    def loadWeightforPredict(self, path):
        model = tf.keras.models.load_model(path)
        predict = model.predict(testImage)
        self.drawPredict(testImage, predict, 0)

    def drawPredict(self, image, predict, index):
        plt.imshow(image[index])
        plt.title(predict[index])
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    image_dir = "./HAM10000_images_test/"
    path = "./test.h5"
    testImage = Predict.Predict_loader(image_dir, 224)
    Predict.loadWeightforPredict(path)
    # drawPredict(testImage, predict, 0)
