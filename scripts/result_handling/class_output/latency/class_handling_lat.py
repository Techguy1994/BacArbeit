import csv
import xml.etree.cElementTree as ET
import argparse
import sys
import os
import pandas as pd

def main():
    api_type, model_name, result_file = handle_arguments()
    result_csv = get_current_general_directory(api_type, result_file, model_name)
    print(result_csv)

    df=pd.read_csv(result_csv)
    inference_time = df["inference time"].astype(float)

    print(inference_time.mean())

    avg = inference_time.mean()
    

    with open(model_name + api_type + ".txt", "w") as f:
        f.write("Avg time: " + str(avg) + "\n")




        

def handle_arguments():
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

    parser.add_argument("-t", "--type", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-f", "--result_file", required=False)
    #parser.add_argument("-s", "--skip_json_creation", required=False, action="store_true")
    args = parser.parse_args()

    return args.type, args.model_name, args.result_file

def get_current_general_directory(type, result_file, model_name):

    if not result_file:
        dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = dir.split("/scripts/result_handling/class_output")[0]
        results_dir = os.path.join(base_dir, "results", "class", type)
        all_models = os.listdir(results_dir)
        print(all_models)
        #sys.exit()

        for model in all_models:
            if model_name == model:

                results_dir = os.path.join(results_dir, model, "output")
        
        print(results_dir)


        all_results_file = os.listdir(results_dir)
        all_results_file_paths = []
                
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
            



if __name__ == "__main__":
    main()