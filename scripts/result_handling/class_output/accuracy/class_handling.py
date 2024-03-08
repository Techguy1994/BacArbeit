import csv
import xml.etree.cElementTree as ET
import argparse
import sys
import os

def main():
    api_type, model_name, result_file = handle_arguments()
    result_csv = get_current_general_directory(api_type, result_file, model_name)
    print(result_csv)
    LOC_synset_mapping = 'LOC_synset_mapping.txt'
    LOC_val_solution = 'LOC_val_solution.csv'

    top_acc_1 = 0
    top_acc_5 = 0
    number_of_images = 0

    solution_dict = {}

    with open(LOC_synset_mapping) as f:
        synset_mapping = list(f.readlines())

    #print(synset_mapping)
    
    with open(LOC_val_solution) as csv_file:
        val_solution = csv.DictReader(csv_file)

        for row in val_solution:
            #print(row)
            solution_dict[row["ImageId"]] = row["PredictionString"].split(" ")[0]
            


    print("--------------------------------------")
    
    with open(result_csv) as csv_file:
        val_result = csv.DictReader(csv_file)

        for res in val_result:

            number_of_images = number_of_images + 1
            
            #print(res)
            #print(res["image"])
            image_name = res["image"].split("/")[-1].split(".JPEG")[0]
            #print(solution_dict[image_name])

            for synset in synset_mapping:
                #print(synset.split(" ")[0])

                if synset.split(" ")[0] == solution_dict[image_name]:
                    solution_label = synset.split(synset.split(" ")[0])[-1]
                    #print("found")
                    break
                    
            temp_dict = {}
            label_string = "label"
            value_string = "value"

            for i in range(5):
                temp_dict[res[label_string + str(i+1)]] = res[value_string + str(i+1)] 
                
            
            #print(solution_label)
            #print(temp_dict)

            biggest_number = float(0.0)

            for label, value in temp_dict.items():
                #print(label, value)
                value = float(value)
                #print(type(value))
                #print(type(biggest_number))
                #print(label, value)

                if float(value) > biggest_number:
                    biggest_number = value


                if label in solution_label:
                    top_acc_5 = top_acc_5 + 1
                    #print(label, value)
                    #print(top_acc_5)

                    if biggest_number == value:
                        #print(biggest_number, value)
                        top_acc_1 = top_acc_1 + 1

            #print(number_of_images, top_acc_1, top_acc_5)
            #sys.exit()
    
    acc_1 = top_acc_1/number_of_images
    acc_5 = top_acc_5/number_of_images

    print(top_acc_1, top_acc_5, acc_1, acc_5)

    with open(model_name + api_type + ".txt", "w") as f:
        f.write("Number of images: " + str(number_of_images) + "\n")
        f.write("Top1 acc: " + str(acc_1) + "\n")
        f.write("Top5 acc: " + str(acc_5) + "\n")




        

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
        base_dir = dir.split("/scripts/result_handling/class_output/accuracy")[0]
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