import argparse
import os
import pandas as pd
import json
import ast
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    type, model_name, result_file, skip = handle_arguments()#
    json_name = type + "_" + model_name + ".json"
    print(type, model_name, result_file)
    result_csv = get_current_general_directory(type, result_file, model_name)
    print(result_csv)
    if not skip:
        get_json_file(result_csv, json_name)
        
    coco_scoring(json_name)



def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

    parser.add_argument("-t", "--type", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-f", "--result_file", required=False)
    parser.add_argument("-s", "--skip_json_creation", required=False, action="store_true")
    args = parser.parse_args()

    return args.type, args.model_name, args.result_file, args.skip_json_creation

def get_current_general_directory(type, result_file, model_name):

    if not result_file:
        dir = os.path.dirname(os.path.realpath(__file__))
        #print("Hello")
        base_dir = dir.split("/scripts/result_handling/detection_output")[0]
        results_dir = os.path.join(base_dir, "results", "det", type)
        all_models = os.listdir(results_dir)

        for model in all_models:
            if model_name in model:
                results_dir = os.path.join(results_dir, model, "output")

        all_results_file = os.listdir(results_dir)
        all_results_file_paths = []
        #print(all_results_file)
        #list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(results_dir, x)), os.listdir(results_dir)))
        #print(list_of_files)
                
        for file in all_results_file:
            if "_Store" in file:
                pass
            else:
                all_results_file_paths.append(os.path.join(results_dir, file))

        latest_file = max(all_results_file_paths, key=os.path.getmtime)
        print(latest_file)
        return latest_file

    else:
        return result_file

def convert_categroy_id_to_coco(category_id_name, ann_categories):

    index = category_id_name.split(" ")[0]
    category_id_name = category_id_name.split(index + " ")[-1]

    for item in ann_categories:
            if category_id_name == item["name"]:
                return item["id"]
            
def get_json_file(csv_file, json_name):
    #print(csv_file)
    #print(os.path.exists(csv_file))
    df = pd.read_csv(csv_file)
    dict_result = df.to_dict()

    json_list = []

    annotations_file = "instances_val2017.json"
    with open(annotations_file, 'r') as a:
        ann = json.load(a)

    ann_categories = ann["categories"]

    for key in dict_result["image"]:
        #print(key)

        image_name = dict_result["image"][key].split("/")[-1]
        image_name = image_name.split(".jpg")[0]
        image_id = int(image_name.lstrip("0"))

        category_id = ast.literal_eval(dict_result["index"][key])
        category_id_name = ast.literal_eval(dict_result["label"][key])
        bbox = ast.literal_eval(dict_result["boxes"][key])
        score = ast.literal_eval(dict_result["value"][key])
        

        for i in range(len(score)):

            category_id = convert_categroy_id_to_coco(category_id_name[i], ann_categories)
            
            #print("cat: ", category_id)

            #print(score[i])
            json_list.append({"image_id": image_id, "category_id": category_id, "bbox": [float(bbox[i][0]), float(bbox[i][1]), float(bbox[i][2]), float(bbox[i][3])], "score": round(score[i],3)})

        with open(json_name, 'w') as f:
            json.dump(json_list, f)
    
def coco_scoring(json_name):
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    print('Running demo for *%s* results.'%(annType))

    annFile = "instances_val2017.json"
    resFile = json_name

    #initialize COCO ground truth api

    #annFile = "annotations_2017/instances_val2017.json"
    cocoGt=COCO(annFile)


    #initialize COCO detections api

    #resFile = resFile%(dataDir, prefix, dataType, annType)
    cocoDt=cocoGt.loadRes(resFile)

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)

    stats = cocoEval.evaluate()
    print("Stats", stats)
    stats = cocoEval.accumulate()
    print("Stats",stats)
    stats = cocoEval.summarize()
    print("Stats",stats)


if __name__ == "__main__":
    main()