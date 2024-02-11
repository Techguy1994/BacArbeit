import yaml
import csv
import cv2
import os
import pandas as pd
import sys
import ast


def main():

    entries = {}

    dir_path = os.path.dirname(os.path.realpath(__file__))
    txt_path = os.path.join(dir_path, "image_txt")
    jpg_path = os.path.join(dir_path, "images")

    #read the yaml file
    with open('coco.yaml', 'r') as file:
        coco = yaml.safe_load(file)

    label_dict = coco["names"]

    df = pd.read_csv("2023_11_8_12_50.csv")
    print(df)

    dict_result = df.to_dict()
    #print(dict_result)
    #print("")


    for key in dict_result["image"]:
        #print(key)
        #print(key, image)

        image_name = dict_result["image"][key].split("/")[-1]
        image_name = image_name.split(".jpg")[0]
        entries[image_name] = []

        #print(len(list(dict_result["index"][key])))
        index = ast.literal_eval(dict_result["index"][key])
        boxes = ast.literal_eval(dict_result["boxes"][key])
        value = ast.literal_eval(dict_result["value"][key])

        for i in range(len(index)):
            entries[image_name].append([index[i], boxes[i], value[i]])

        #print(dict_result["index"][key])
        #print(dict_result["boxes"][key])
        #print(dict_result["value"][key])



    #for _, index in dict_result["index"].items():
    #    print(index)

    #for _, boxes in dict_result["boxes"].items():
    #    print(boxes)

    print(entries)

    #for index, _ in label_dict.items():
    #    print("label: ", index)
    
    for x, y in label_dict.items():
        print("label: ", x)
        if x == 75:

            total_instances = 0

            for image, result in entries.items():
                jpg = cv2.imread(os.path.join(jpg_path, image + ".jpg"))
                w, h = jpg.shape[1], jpg.shape[0]
                
                with open(os.path.join(txt_path, image + ".txt")) as f:
                    lines = f.readlines()
                    for line in lines:
                        #print(line)
                        elements = line.split(" ")
                        elements[-1] = elements[-1].split("\n")[0]
                        
                        if int(elements[0]) == x:
                            total_instances = total_instances + 1
                            print(elements) 

                            for res in result:
                                pass
                                

                            

                    print("end")
            print(total_instances)


    sys.exit()

    with open("2023_11_8_12_50.csv") as csv_file:
        csv_reader = csv.reader(csv_file)

        entries = {}

        for row in csv_reader:
            if row[0] == "":
                continue

            image = row[1].split("/")[-1]
            image_name = image.split(".jpg")[0]
            entries[image_name] = []
            print(row[4])
            l = list(row[4][0])
            print(l)
            #print(int(row[4]))
            #print(len(list(row[4])))
            for i in range(len(row[4])):
                entries[image_name].append([row[4][i], row[6][i], row[5][i]])
            
            break

        print(entries)



        

        #image = row[1].split("/")[-1]
        #image_name = image.split(".jpg")[0]

        #with open(os.path.join(txt_path, image_name + ".txt")) as f:
        #    lines = f.readlines()

        #    for line in lines:
                #print(line)
        #        pass

                #print(image_name)
        
    """
    with open('000000000139.txt') as f:
        lines = f.readlines()

        output_img = cv2.imread("000000000139.jpg")

        for line in lines:
            print(line)
            list = line.split(" ")
            list[-1] = list[-1].split("\n")[0]
            list = [float(i) for i in list]
            print(list)

            x_min = int((list[1] - list[3]/2)*640)
            y_min = int((list[2] - list[4]/2)*426)

            x_max = int((list[1] + list[3]/2)*640)
            y_max = int((list[2] + list[4]/2)*426)

            print(x_min, y_min, x_max, y_max)

            

            output_img = cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)

        cv2.imwrite("output.jpg", output_img)
    """



    #read the csv file 
    #with open("2023_11_8_12_50.csv") as csv_file:
    #    csv_reader = csv.reader(csv_file)

    #for row in csv_reader:
    #    print(row)

def iou(box1, box2):
    """
    Calculates the intersection-over-union (IoU) value for two bounding boxes.

    Args:
        box1: Array of positions for first bounding box
              in the form [x_min, y_min, x_max, y_max].
        box2: Array of positions for second bounding box.

    Returns:
        Calculated intersection-over-union (IoU) value for two bounding boxes.
    """
    print(box1)
    print(box2)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_box1 <= 0 or area_box2 <= 0:
        iou_value = 0
    else:
        y_min_intersection = max(box1[1], box2[1])
        x_min_intersection = max(box1[0], box2[0])
        y_max_intersection = min(box1[3], box2[3])
        x_max_intersection = min(box1[2], box2[2])

        area_intersection = max(0, y_max_intersection - y_min_intersection) *\
                            max(0, x_max_intersection - x_min_intersection)
        area_union = area_box1 + area_box2 - area_intersection

        try:
            iou_value = area_intersection / area_union
        except ZeroDivisionError:
            iou_value = 0

    return iou_value



if __name__ == "__main__":
    main()