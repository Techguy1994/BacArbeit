import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np

def main():
    args = handle_arguments()
    if args.new:
        create_new_database(args.database)
    elif args.update:
        df = manage_database(args)
        df = df.loc[:, ~df.columns.str.match('Unnamed')]
        df.to_csv(args.database, index=False)
        
        #print(df)
    elif args.read:
        df = load_database(args.database)
        #print(df["latency"])

def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

    parser.add_argument("-n", "--new", required=False, help="create new database with giving name", action="store_true")
    parser.add_argument("-u", "--update", required=False, help="add or update line of the database", action="store_true")
    parser.add_argument("-d", "--database", required=False, default="database.csv")
    parser.add_argument("-r", "--read", required=False, action="store_true")
    parser.add_argument("-t", "--type", required=False)
    parser.add_argument("-api", "--api", required=False)
    parser.add_argument("-m", "--model", required=False)
    parser.add_argument("-f", "--file_name", required=False)
    parser.add_argument("-os", "--os", required=False, default="ubuntus")

    return parser.parse_args()


def create_new_database(database_name):
    dict = {
        "model_name": [],
        "os": [],
        "api": [],
        "inference type": [],
        "latency avg": [],
        "top1": [],
        "top5": [],
        "map": [],
        "miou": [],
        "latency": [],
        "thread count": [],
        "load": []
    }

    df = pd.DataFrame(dict)

    print(df)

    df.to_csv(database_name, index=False)

def manage_database(args):
    df = load_database(args.database)
    print(args.file_name)

    d = "/Users/marounel-chayeb/BacArbeit/final_results/results/det/stress-ng/"
    apis = ["ov", "tf", "onnx", "delegate", "pytorch"]

    load_directories = os.listdir(d)

    print(load_directories)
    

    load_directories= [item for item in load_directories if not item.startswith('.')]
    print(load_directories)
    


    for l in load_directories:
        print(d,l)
        load_directory = os.path.join(d, l)

        path = load_directory 
        csv_files = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        print(csv_files)
        print("------")
        for file_path in csv_files:
            file_name = file_path.split("/")[-1]
            file_api =  file_path.split("/")[-2]

            print(file)

            is_only_lat = file_name.split("_")[-1]
            if "onlylat" in is_only_lat:
                ops = file_name.split("_")[-3]
                thread = file_name.split("_")[-2]
                inference_type = file_path.split("/")[-4]
                model_name = file_path.split("/")[-2]
                api = file_api

                print(ops, inference_type, model_name, api, thread)
    
            if df.empty:
                print("empty")
                df = add_entry(model_name, inference_type, api, is_only_lat, ops, file_path, df, thread, l)
            else:
                print("----")
                print(file_path)
                
                
                ind = df.index[(df["model_name"] == model_name) & (df["os"] == ops) & (df["api"] == api) & (df["thread count"] == str(thread)) & (df["load"] == l)]
               
                print(ind)
                if len(ind) == 0:
                    print("no row found")
                    df = add_entry(model_name, inference_type, api, is_only_lat, ops, file_path, df, thread, l)
                else:
                    filt = ((df["model_name"] == model_name) & (df["api"] == api)) & (df["load"] == l) & (df["thread count"] == str(thread))

                    if "onlylat" in is_only_lat:
                        print("found")
                        print(file_path)
                        print(np.mean(get_latency_value(file_path)))
                        

                        df.loc[filt, "os"] = ops
                        df.loc[filt, "latency"] = file_path
                        df.loc[filt, "latency avg"] = np.mean(get_latency_value(file_path))

                        print(df[filt])
                    

                        #return df
                    
    return df
                    


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def load_database(database_name):
    df = pd.read_csv(database_name)
    return df

def add_entry(model_name, inference_type, api, onyl_lat, ops, file_path, df, thread, l):
    if thread == "False":
        entry = {"model_name": [model_name], "api": [api], "inference type": [inference_type], "os": [ops], "load": [l], "thread count": [thread]}
    else:
        entry = {"model_name": [model_name], "api": [api], "inference type": [inference_type], "os": [ops], "load": [l], "thread count": [str(int(thread))]}
    if "onlylat" in onyl_lat:
        #inference_time = df["inference time"].astype(float)
        #avg = inference_time.mean()
        lat = get_latency_value(file_path)
        print(file_path)
        entry.update({"latency": file_path, "latency avg": np.mean(lat)})
        #new_row = pd.DataFrame(entry)
        #df = df.append(new_row, index=df.columns)
        df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)

        return df
    else: 
        if inference_type == "class":
            print(file_path)

            with open(file_path, "r") as f:
                content = f.readlines()
                print(content)
                top1 = content[1].split("acc: ")[-1].split("\n")[0]
                top5 = content[2].split("acc: ")[-1].split("\n")[0]
            print(top1, top5)
            entry.update({"top1": [top1], "top5": [top5]})
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)

            return df
        elif inference_type == "det":
            print(inference_type)

            with open(file_path, mode="r") as det_csv:
                det_dict = csv.DictReader(det_csv, delimiter=";")
                #for row in det_dict:
                #    print(row)

                for row in det_dict:
                    print(api)
                    if row["Type"] == api:
                        print(row["model_name"])
                        if row["model_name"] == model_name:
                            map = row["Map (IoU=0.50:0.95)"]
                            map = float(map.replace(",", "."))
            
            entry.update({"map": [map]})
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)
            print(df)

            return df

        elif inference_type == "seg":
            print(inference_type)

            with open(file_path, mode="r") as seg_csv:
                seg_dict = csv.DictReader(seg_csv, delimiter=";")
                #for row in det_dict:
                #    print(row)

                for row in seg_dict:
                    print(api)
                    if row["Type"] == api:
                        print(row["model_name"])
                        if row["model_name"] == model_name:
                            miou = float(row["Accuracy"].split("%")[0])
                            
            entry.update({"miou": [miou]})
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)
            print(df)
            return df


def update_entry():
    pass

def get_latency_value(csv_path):
    print(csv_path)

    res = pd.read_csv(csv_path)
    return res["inference time"].to_list()

def find_all_models():
    pass

def find_classification_information():
    pass

def find_detection_information():
    pass

def find_segmentation_information():
    pass

if __name__ == "__main__":
    main()