import pandas as pd
from datetime import datetime
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
    }

    return data

def store_output_dictionary_det(data, image, time, output):
    pass

def create_base_dictionar_seg():
    pass  

def store_output_dictionary_seg():
    pass

def create_pandas_dataframe(dict):
    df = pd.DataFrame(dict)
    return df

def store_pandas_data_frame_as_csv(df, output):
    date = datetime.now()
    file_name = str(date.year)+ "_" + str(date.month) + "_" + str(date.day) + "_" + str(date.hour) + "_" + str(date.minute) + ".csv"
    location = os.path.join(output, "output", file_name)
    df.to_csv(location)

def store_pandas_data_frame_as_csv_det_seg(df, output, name_date):
    location = os.path.join(output, "output", name_date + ".csv")
    df.to_csv(location)




    