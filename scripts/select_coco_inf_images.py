import subprocess
import argparse
import sys
import json
import os 
import shutil

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

    parser.add_argument("-j", "--input_json", required=True)
    args = parser.parse_args()

    input_json = args.input_json
    list_of_images = []

    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.split("/scripts")[0]
    output_path = os.path.join(dir_path, "coco_test_images")
    input_path = os.path.join(dir_path, "cocoval2017")
    print(output_path)
    print(input_path)

    with open(input_json) as ann:
        annotations_data = json.load(ann)

    for images in annotations_data["images"]:
        #list_of_images.append(images["file_name"])
        src = os.path.join(input_path, images["file_name"])
        shutil.copy(src, output_path)










    

if __name__ == "__main__":
    main()
