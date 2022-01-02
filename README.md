# DesignPattern_FinalProject
110-1 SoftWare Design - Desing Pattern Refactor final project

This task is to reconstruct the code. The original file source is:
https://github.com/ac005sheekar?tab=repositories  Thank you so much!

Dataset URL: https://challenge.isic-archive.com/data/#2018
CSV: ./HAM10000_metadata.csv

***You need to manually merge the following two folders into one first
"HAM10000_images_part1" and "HAM10000_images_part2" synthesize "HAM10000_images"

Execute command string:
$ python main.py -i "./HAM10000_images/" -c "./HAM10000_metadata.csv" -r 224 -tr 0.2 -cl 7 -m ResNet50 -p True -s test.h5

Instruction:

-image_dir = "./HAM10000_images/"

-metaDataCSV = "./HAM10000_metadata.csv"

-resize = 224

-test_radio = 0.2

-classNum = 7

-modelName = "vgg16", "resnet50" or "efnb0"

-pretrained = True or False

-saveWeightName = "test.h5"
