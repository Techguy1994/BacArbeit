from datetime import datetime
import pandas as pd
import os 

def create_base_dictionary_class(n_big):
    data = {
        "image": [],
        "inference time": []
    }

    for i in range(n_big):
        data["label" + str(i+1)] = []
        data["index" + str(i+1)] = []
        data["value" + str(i+1)] = []

    return data 

def store_output_dictionary_class(data, image, time, output, n_big):
    data["image"].append(image)
    data["inference time"].append(time)

    for i in range(n_big):
        data["label" + str(i+1)].append(output[i]["label"])
        data["index" + str(i+1)].append(output[i]["index"])
        data["value" + str(i+1)].append(output[i]["value"])

    return data

def create_base_dictionary_det():
    data = {
        "image": [],
        "inference time": [],
        "label": [],
        "index": [],
        "value": [],
        "boxes": []
    }

    return data

def store_output_dictionary_det(data, image, time, output):
    data["image"].append(image)
    data["inference time"].append(time)
    data["label"].append([])
    data["index"].append([])
    data["value"].append([])
    data["boxes"].append([])


    for i in range(len(output)):
        data["label"][-1].append(output[i]["label"])
        data["index"][-1].append(output[i]["index"])
        data["value"][-1].append(output[i]["value"])
        data["boxes"][-1].append(output[i]["boxes"])
    
    return data

def create_base_dictionary_seg():
    data = {
        "image": [],
        "inference time": [],
        "label": []
    }  

    return data

def store_output_dictionary_seg(data, image, time, output):
    data["image"].append(image)
    data["inference time"].append(time)
    data["label"].append([])

    #print(output)
    
    for i in range(len(output)):
        #print(data["label"][-1])
        #print(output[i])
        data["label"][-1].append(output[i])

    return data

def store_output_dictionary_only_lat(data, image, time, n_big):
    data["image"].append(image)
    data["inference time"].append(time)

    for i in range(n_big):
        data["label" + str(i+1)].append(False)
        data["index" + str(i+1)].append(False)
        data["value" + str(i+1)].append(False)

    return data

def store_output_dictionary_det_only_lat(data, image, time):
    data["image"].append(image)
    data["inference time"].append(time)
    data["label"].append([])
    data["index"].append([])
    data["value"].append([])


    #for i in range(len(output)):
    #    data["label"][-1].append(output[i]["label"])
    #    data["index"][-1].append(output[i]["index"])
    #    data["value"][-1].append(output[i]["value"])
    
    return data

def store_output_dictionary_seg_only_lat(data, image, time):
    data["image"].append(image)
    data["inference time"].append(time)
    data["label"].append([])

    
    #for i in range(len(output)):
    #    print(data["label"][-1])
    #    print(output[i])
    #    data["label"][-1].append(output[i])

    return data



def create_pandas_dataframe(dict):
    #print(dict)
    df = pd.DataFrame(dict)
    return df

def store_pandas_data_frame_as_csv(df, output, name_date, type, model_name, api):
    model_name = model_name.split("/")[-1]
    file_name = model_name + "_" + api + "_" + type + "_" + name_date + ".csv"
    location = os.path.join(output, "output", file_name)
    print(location)
    df.to_csv(location)

def store_pandas_data_frame_as_csv_det_seg(df, output, name_date, type, model_name, api):
    model_name = model_name.split("/")[-1]
    file_name = model_name + "_" + api + "_" + type + "_" + name_date + ".csv"
    location = os.path.join(output, "output", file_name)
    df.to_csv(location)




    