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

        if args.del_ov_class:
            print("delete")
            df = delete_custom_rows(args)
        else:
            print("update")
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
    parser.add_argument("-del", "--del_ov_class", required=False, action="store_true")

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
        "latency": []
    }

    df = pd.DataFrame(dict)

    print(df)

    df.to_csv(database_name, index=False)

def manage_database(args):
    df = load_database(args.database)

    file_path = find(args.file_name, "/Users/marounel-chayeb/BacArbeit/final_results/results/")

    is_only_lat = args.file_name.split("_")[-1]
    if "onlylat" in is_only_lat:
        print("---")
        os = args.file_name.split("_")[-3]
        inference_type = file_path.split("/")[-4]
        model_name = file_path.split("/")[-2]
        api = file_path.split("/")[-3]
    else:
        os = args.os
        #os = "Nan"

        print(file_path)
        inference_type = file_path.split("/")[-4]

        if inference_type == "class":
            model_name = file_path.split("/")[-2]
            api = file_path.split("/")[-3]
        else:
            model_name = args.model
            api = args.api
            inference_type = args.type
            print(file_path, api, model_name, inference_type)
    



    

    
    if df.empty:
        df = add_entry(model_name, inference_type, api, is_only_lat, os, file_path, df)
        return df
    else:

        #print(model_name)
        #print(df.columns)
        #print(df)
        
        ind = df.index[(df["model_name"] == model_name) & (df["os"] == os) & (df["api"] == api)]
        if len(ind) == 0:
            #print("no row found")
            df = add_entry(model_name, inference_type, api, is_only_lat, os, file_path, df)
            return df
        else:

            print(model_name, api, os)
            filt = (df["model_name"] == model_name) & (df["api"] == api) & (df["os"] == os)

            if "onlylat" in is_only_lat:

                df.loc[filt, "os"] = os
                df.loc[filt, "latency"] = file_path
                df.loc[filt, "latency avg"] = np.mean(get_latency_value(file_path))

                return df
            else:
                if inference_type == "class":
                    with open(file_path, "r") as f:
                        content = f.readlines()
                        #print(content)
                        top1 = float(content[1].split("acc: ")[-1].split("\n")[0])
                        top5 = float(content[2].split("acc: ")[-1].split("\n")[0])

                    df.loc[filt, "top1"] = float(top1)
                    df.loc[filt, "top5"] = float(top5)

                    return df
                elif inference_type == "det":
                    with open(file_path, mode="r") as det_csv:
                        det_dict = csv.DictReader(det_csv, delimiter=";")
                        #for row in det_dict:
                        #    print(row)

                        for row in det_dict:
                            #print(api)
                            if row["Type"] == api:
                                #print(row["model_name"])
                                if row["model_name"] == model_name:
                                    map = row["Map (IoU=0.50:0.95)"]
                                    map = float(map.replace(",", "."))
                    
                    df.loc[filt, "map"] = map

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

                    df.loc[filt, "miou"] = miou

                    return df
                
        #print(df.iloc[ind])
        sys.exit()
    #print(model_name)
    #print(api)
    #print(inference_type)
    #print(file_csv_path)

    #index = df.index(df[model_name])
    #print(index)

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def load_database(database_name):
    df = pd.read_csv(database_name)
    return df

def add_entry(model_name, inference_type, api, onyl_lat, os, file_path, df):
    entry = {"model_name": [model_name], "api": [api], "inference type": [inference_type], "os": [os]}
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
                    print("-")
                    print(api, row["Type"])
                    if row["Type"] == api:
                        
                        print("--")
                        print(row["model_name"].strip())
                        if row["model_name"].strip() == model_name:
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
                    #print(api)
                    if row["Type"] == api:
                        print(row["model_name"])
                        if row["model_name"] == model_name:
                            miou = float(row["Accuracy"].split("%")[0])
                            print(miou)
                
                            
            entry.update({"miou": [miou]})
            df = pd.concat([df, pd.DataFrame(entry)], ignore_index=True)
            print(df)
            return df



def get_latency_value(csv_path):
    print(csv_path)

    res = pd.read_csv(csv_path)
    return res["inference time"].to_list()

def delete_custom_rows(args):
    df = load_database(args.database)

    i = df[((df.loc[:, "model_name"] == "mobilenet_v3_large_q") | (df.loc[:, "model_name"] == "mobilenet_v2_q"))].index


    df = df.drop(i)

    return df

if __name__ == "__main__":
    main()