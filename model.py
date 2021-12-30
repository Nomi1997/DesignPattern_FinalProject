from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import *
from tensorflow.keras.models import *

class modelTemplate():
    def getPretrainedName(self, modelName):
        if modelName == "resnet50":
            modelName = ResNet50
        elif modelName == "vgg16":
            modelName = VGG16
        
        return modelName

    def finalDense(self, input, classNum, activation):
        finalLayer = Dense(classNum, activation = activation)(input)
        return finalLayer

    def generatePretrained(self, input, modelType):
        modelType = self.getPretrainedName(modelType)
        model = modelType(include_top=False, weights='imagenet', input_tensor=None, input_shape=(input[0], input[1], 3))
        model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        return model

class VGG16Custom(modelTemplate):

    def doubleConv2D(self, input, channel, kernel = (3, 3), stride = (1, 1), activation = 'relu', padding = 'same'):
        x = Conv2D(channel, kernel, stride, padding, activation = activation)(input)
        x = Conv2D(channel, kernel, stride, padding, activation = activation)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x

    def tripleConv2D(self, input, channel, kernel = (3, 3), stride = (1, 1), activation = 'relu', padding = 'same'):
        x = Conv2D(channel, kernel, stride, padding, activation = activation)(input)
        x = Conv2D(channel, kernel, stride, padding, activation = activation)(x)
        x = Conv2D(channel, kernel, stride, padding, activation = activation)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x

    def multiDense(self, input, activation):
        x = Flatten()(input)
        x = Dense(4096, activation = activation)(x)
        x = Dense(4096, activation = activation)(x)
        return x

    def generateModel(self, input, channel, classNum, kernel = (3, 3), stride = (1, 1), activation = 'relu', padding = 'same'):
        x = self.doubleConv2D(input, channel, kernel, stride, activation, padding)
        x = self.doubleConv2D(x, channel * 2, kernel, stride, activation, padding)
        x = self.tripleConv2D(x, channel * 4, kernel, stride, activation, padding)
        x = self.tripleConv2D(x, channel * 8, kernel, stride, activation, padding)
        x = self.multiDense(x, activation)
        output = self.finalDense(x, classNum, 'softmax')

        model = Model(inputs = input, outputs = output, name='vgg16')
        model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        return model

class ResNetCustom(modelTemplate):
    def bottleNeck(self, input, channel, kernel = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same'):
        x = Conv2D(channel, 1, strides, padding, activation = activation)(input)
        shortcut = x
        x = BatchNormalization()(x)
        x = Conv2D(channel, kernel, strides, padding, activation = activation)(x)
        x = BatchNormalization()(x)
        x = Conv2D(channel * 4, 1, strides, padding, activation = activation)(x)
        x = Concatenate()([x, shortcut])
        return x

    def layerState(self, input, channel, layerNums):
        for i in range(layerNums):
            input = self.bottleNeck(input, channel)
        return input

    def generateModel(self, input, channel, classNum, layerNums):
        x = Conv2D(channel, 1, strides = (1, 1) , padding = 'same', activation = 'relu')(input)
        x = BatchNormalization()(x)
        x = self.layerState(x, channel, layerNums[0])
        x = self.layerState(x, channel * 2, layerNums[1])
        x = self.layerState(x, channel * 4, layerNums[2])
        x = self.layerState(x, channel * 8, layerNums[3])
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        output = self.finalDense(x, classNum, 'softmax')

        model = Model(inputs = input, outputs = output, name='resnet50')
        model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        return model

# class EfficientNetCustom(modelTemplate):

if __name__ == '__main__':
    input_size = (256,256,3)

    # model = VGG16Custom()
    # model = model.generatePretrained([256,256,3], VGG16)
    # model = model.generateModel(Input(input_size), 64, 7)

    model = ResNetCustom()
    # model = model.generatePretrained([256,256,3], ResNet50)
    model = model.generateModel(Input(input_size), 64, 7, [3, 4, 6, 3])

    model.summary()   
