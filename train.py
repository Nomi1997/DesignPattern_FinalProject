from model import *
from dataloader import *
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

class Train():
    def __init__(self, image_dir, metaDataCSV, resize, test_radio, SEED, classNum, modelName, pretraind, saveWeightName):
        self.image_dir = image_dir
        self.label_dir = metaDataCSV
        self.resize = resize
        self.test_radio = test_radio
        self.SEED = SEED
        self.label_data = self.csvInfo(metaDataCSV)
        self.modelName = modelName
        self.pretraind = pretraind
        self.saveWeightName = saveWeightName
        self.classNum = classNum
        self.red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.7)
        self.datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.1, # Randomly zoom image
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

    def csvInfo(self, label_dir):
        metaDataCSVInfo = pd.read_csv(label_dir, encoding='utf-8')
        uni = sorted(metaDataCSVInfo['dx'].unique())
        for i in range(len(uni)):
            metaDataCSVInfo['dx'].replace(uni[i], i , inplace=True)
        
        return metaDataCSVInfo[['image_id', 'dx']]

    def getData(self, image_dir, label_data, resize, test_radio, SEED):
        img_list, lab_list = Dataset_loader(image_dir, label_data, resize)
        train_inputs, val_inputs, train_label, val_label = train_test_split(img_list, lab_list, test_size = test_radio, random_state = SEED)

        return train_inputs, val_inputs, train_label, val_label
    
    def trainState(self, batch_size, epochs, isSave = True, isDraw = False):
        train_inputs, val_inputs, train_label, val_label = self.getData(self.image_dir, self.label_data, self.resize, self.test_radio, self.SEED)
        
        input_size = [self.resize, self.resize, 3]
        if self.pretraind:
            getModel = modelTemplate()
            model = getModel.generatePretrained(input_size, self.modelName)
        else:
            if self.modelName == "resnet50":
                getModel = ResNetCustom()
                model = getModel.generateModel(Input(shape = input_size), 64, self.classNum)
            elif self.modelName == "vgg16":
                getModel = VGG16Custom()
                model = getModel.generateModel(Input(shape = input_size), 64, self.classNum)

        History = model.fit_generator(self.datagen.flow(train_inputs,train_label,batch_size=batch_size),validation_data=(val_inputs,val_label),
                              epochs= epochs, steps_per_epoch=train_inputs.shape[0]//batch_size,verbose=1,callbacks=[self.red_lr])
        
        if isSave: 
            model.save_weights(self.saveWeightName)
            model_json = model.to_json()
            with open(self.saveWeightName[:-3] + ".json", "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")
        if isDraw: self.drawAcc(History)

    def drawAcc(self, History):
        plt.plot(History.history['acc'])
        plt.plot(History.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'test'])
        plt.show()
