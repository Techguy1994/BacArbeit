#import yaml
#import csv
#import cv2
import os
import pandas as pd
import sys
import ast
import json


def main():

    entries = {}

    dir_path = os.path.dirname(os.path.realpath(__file__))
    txt_path = os.path.join(dir_path, "image_txt")
    jpg_path = os.path.join(dir_path, "images")

    #rea the yaml file
    #with open('coco.yaml', 'r') as file:
    #   coco = yaml.safe_load(file)

    #label_dict = coco["names"]

    df = pd.read_csv("2024_1_25_21_32.csv")
    print(df)

    dict_result = df.to_dict()
    #print(dict_result)
    #print("")

    json_list = []


    for key in dict_result["image"]:

        #image_id
        image_name = dict_result["image"][key].split("/")[-1]
        image_name = image_name.split(".jpg")[0]
        print(image_name)
        #print(image_name.split("0"))
        print(image_name.lstrip("0"))
        image_id = int(image_name.lstrip("0"))
        

        category_id = ast.literal_eval(dict_result["index"][key])
        bbox = ast.literal_eval(dict_result["boxes"][key])
        score = ast.literal_eval(dict_result["value"][key])

        for i in range(len(category_id)):
            json_list.append({"image_id": image_id, "category_id": category_id[i], "bbox": bbox[i], "score": round(score[i],3)})

        #print(len(list(dict_result["index"][key])))
        #index = ast.literal_eval(dict_result["index"][key])
        #boxes = ast.literal_eval(dict_result["boxes"][key])
        #value = ast.literal_eval(dict_result["value"][key])

        #for i in range(len(index)):
        #    entries[image_name].append([index[i], boxes[i], value[i]])

        #print(dict_result["index"][key])
        #print(dict_result["boxes"][key])
        #print(dict_result["value"][key])

        with open('data.json', 'w') as f:
            json.dump(json_list, f)









if __name__ == "__main__":
    main()