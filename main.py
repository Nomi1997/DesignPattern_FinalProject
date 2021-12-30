import argparse
from train import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Enter the image folder address.")
    parser.add_argument("-c", "--metaDataCSV", type=str, required=True, help="Enter the address of the CSV file.")
    parser.add_argument("-r", "--resize", type=int, default=224, required=True, help="Enter the changed image size.")
    parser.add_argument("-tr", "--test_radio", type=float, default=0.2, help="Enter the test ratio.")
    parser.add_argument("-cl", "--classNum", type=int, default=7, required=True, help="Enter the number of categories.")
    parser.add_argument("-m", "--modelName", type=str, required=True, help="Enter the model name.")
    parser.add_argument("-p", "--pretraind", type=bool, default=False, help="Enter whether you want to use a pre-trained model.")
    parser.add_argument("-s", "--saveWeightName", type=str, required=True, help="Enter the name of the weight to be saved.")
    
    args = parser.parse_args()

    image_dir = args.image_dir
    metaDataCSV = args.metaDataCSV
    resize = args.resize
    test_radio = args.test_radio
    classNum = args.classNum
    modelName = args.modelName
    pretraind = args.pretraind
    saveWeightName = args.saveWeightName

    Train(image_dir, metaDataCSV, resize, test_radio, 42, classNum, modelName, pretraind, saveWeightName)

if __name__ == '__main__':
    main()
    # Enter args in terminal:
    # python main.py -i "./HAM10000_images/" -c "./HAM10000_metadata.csv" -r 224 -tr 0.2 -cl 7 -m ResNet50 -p True -s test.h5
