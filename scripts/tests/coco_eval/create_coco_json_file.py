#import yaml
#import csv
#import cv2
import os
import pandas as pd
import sys
import ast
import json


def convert_categroy_id_to_coco(category_id_name, ann_categories):

        #print(category_id_name)

        index = category_id_name.split(" ")[0]
        #print(index)
        category_id_name = category_id_name.split(index + " ")[-1]

        #print(category_id_name)

        for item in ann_categories:
             if category_id_name == item["name"]:
                  return item["id"]

def main():



    dir_path = os.path.dirname(os.path.realpath(__file__))



    df = pd.read_csv("2024_2_10_22_30_ov_alt.csv")


    dict_result = df.to_dict()

    json_list = []

    annotations = "instances_val2017.json"

    with open(annotations, 'r') as a:
        ann = json.load(a)

    ann_categories = ann["categories"]

    for key in dict_result["image"]:

        image_name = dict_result["image"][key].split("/")[-1]
        image_name = image_name.split(".jpg")[0]
        image_id = int(image_name.lstrip("0"))

        category_id = ast.literal_eval(dict_result["index"][key])
        category_id_name = ast.literal_eval(dict_result["label"][key])
        bbox = ast.literal_eval(dict_result["boxes"][key])
        score = ast.literal_eval(dict_result["value"][key])
        

        for i in range(len(score)):

            category_id = convert_categroy_id_to_coco(category_id_name[i], ann_categories)
            print(i)
            #print("cat: ", category_id)

            #print(score[i])
            json_list.append({"image_id": image_id, "category_id": category_id, "bbox": [float(bbox[i][0]), float(bbox[i][1]), float(bbox[i][2]), float(bbox[i][3])], "score": round(score[i],3)})


        with open('data_full_025_045_ov_alt.json', 'w') as f:
            json.dump(json_list, f)



if __name__ == "__main__":
    main()