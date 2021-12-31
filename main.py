import argparse
from train import *
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Enter the image folder address.")
    parser.add_argument("-c", "--metaDataCSV", type=str, required=True, help="Enter the address of the CSV file.")
    parser.add_argument("-r", "--resize", type=int, default=224, required=True, help="Enter the changed image size.")
    parser.add_argument("-tr", "--test_radio", type=float, default=0.2, help="Enter the test ratio.")
    parser.add_argument("-cl", "--classNum", type=int, default=7, required=True, help="Enter the number of categories.")
    parser.add_argument("-m", "--modelName", type=str, required=True, help="Enter the model name.")
    parser.add_argument("-p", "--pretrained", type=bool, default=True, help="Enter whether you want to use a pre-trained model.")
    parser.add_argument("-s", "--saveWeightName", type=str, required=True, help="Enter the name of the weight to be saved.")
    
    args = parser.parse_args()

    # Only these three models can be used, which can be added later
    accessModel = ["resnet50", "vgg16", "efnb0"]
    if args.modelName not in accessModel:
        parser.error("please enter the model name again!")
    
    # Restrict EfficientNet can only use pretrained model
    if args.modelName == "efnb0" and args.pretrained == False:
        parser.error("The EfficientNet series currently only provides pretrained methods.")

    image_dir = args.image_dir
    metaDataCSV = args.metaDataCSV
    resize = args.resize
    test_radio = args.test_radio
    classNum = args.classNum
    modelName = args.modelName
    pretrained = args.pretrained
    saveWeightName = args.saveWeightName

    # image_dir = "./HAM10000_images/"
    # metaDataCSV = "./HAM10000_metadata.csv"
    # resize = 224
    # test_radio = 0.2
    # classNum = 7
    # modelName = "vgg16"
    # pretrained = False
    # saveWeightName = "test.h5"
    
    train = Train(image_dir, metaDataCSV, resize, test_radio, 42, classNum, modelName, pretrained, saveWeightName)
    train.trainState(batch_size = 64, epochs = 150, isSave = True, isDraw = False)

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable = True)
    main()
    # Enter args in terminal:
    # python main.py -i "./HAM10000_images/" -c "./HAM10000_metadata.csv" -r 224 -tr 0.2 -cl 7 -m ResNet50 -p True -s test.h5
