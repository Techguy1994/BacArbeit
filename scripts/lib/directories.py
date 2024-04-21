import os 
import sys

def search_file(file_name, directory):

    for dirpath, dirnames, files in os.walk(directory):

        for file in files:
            if file == file_name:
                return os.path.join(dirpath, file_name)

    sys.exit(f"Error: The file {file_name} does not exist in the directory {directory}!")


def handle_model_directory(args):

    if args.api == "pytorch":
        if args.model:
            if ".pt" in args.model:
                general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
                general_model_dir = os.path.join(general_dir, "models")
                return search_file(args.model, general_model_dir)
            return args.model
        elif args.model_path:
            return args.model_path

    if args.model_path:
       if os.path.exists(args.model_path):
            return args.model_path
       else:
           sys.exit("Error: The model file path given does not exit! Please enter a vaild path or give the path name")
    elif args.model:
        general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
        general_model_dir = os.path.join(general_dir, "models")
        return search_file(args.model, general_model_dir)
    else:
        sys.exit("Error: no model name or model path given. Please Enter with -m a model name or with -mp the model path")

def handle_image_directory(args):
    image_list = []

    if args.randomized_input:
        image_list = create_random_input(args.randomized_input)
        return image_list

    if args.image_folder:
        if os.path.exists(args.image_folder):
            for img in os.listdir(args.image_folder):
                if ".jpg" in img or ".JPEG" in img or ".png" in img and "._" not in img:
                    image_list.append(os.path.join(args.image_folder, img))
        else:
            sys.exit("Error: The image folder path given does not exit! Please enter a vaild path or give the path name")
    elif args.image:
        if os.path.exists(args.image):
            image_list.append(args.image)
    elif args.image_folder_default:
        general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
        general_image_dir = os.path.join(general_dir, "images")
        for img in os.listdir(general_image_dir):
                if ".jpg" in img or ".JPEG" in img or ".png" in img and "._" not in img:
                    image_list.append(os.path.join(general_image_dir, img))   
    else:
        sys.exit("Error: No image option chosen as input. You can either give the image with -img , image folder path with -imgp or choose the default image folder with -imgd!")
    
    return image_list

def create_random_input(count):

    import numpy as np
    from PIL import Image

    image_list = []

    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
    randomized_image_folder = os.path.join(general_dir, "randomized_iamges")
    if not os.path.exists(randomized_image_folder):
        os.makedirs(randomized_image_folder) 

    for i in range(int(count)):
        output = np.random.randint(255, size=(500, 500, 3), dtype=np.uint8)
        image = Image.fromarray(output)
        file_name = os.path.join(randomized_image_folder, str(i) + ".jpg")
        image.save(file_name)
        image_list.append(file_name)

    return image_list

def handle_label_directory(args):
    #print(args)
    if args.labels:
        general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
        general_labels_dir = os.path.join(general_dir, "labels")
        labels_path = os.path.join(general_labels_dir, args.labels)
        if os.path.exists(labels_path):
            return labels_path
        else:
            sys.exit("The label file name given does not exit! Please enter a vaild path or give the path name")
    elif args.labels_path:
        if os.path.exists(args.labels_path):
            return args.labels_path
        else: 
            sys.exit("The label file name given does not exit! Please enter a vaild path or give the path name")
    else: 
         sys.exit("Error: No label name or path given. Please Enter with -l a model name or with -lp the model path")

def handle_output_directory(args, api, type, model, profiler):
    general_dir = os.path.abspath(os.path.dirname(__file__)).split("scripts")[0]
    general_outputs_dir = os.path.join(general_dir, "results")

    if args.output:
        return args.output
    elif args.output_path:
        return 2
    elif args.output_default:
        #handle type 
        outputs_dir = os.path.join(general_outputs_dir, type, api)
        if api == "ov":
            model_name = model.split("/")[-1].split(".xml")[0]
        else:
            model_name = model.split("/")[-1].split(".")[0]

        outputs_dir = os.path.join(outputs_dir, model_name)

        output = os.path.join(outputs_dir, "output")

        if profiler == "perfcounter":
            time_dir = os.path.join(outputs_dir, "perfcounter")

        if profiler == "cprofiler":
            time_dir = os.path.join(outputs_dir, "cprofiler")

        if type in ["det", "seg"]:
            image_dir = os.path.join(outputs_dir, "images")

            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

        if not os.path.exists(output):
            os.makedirs(output)

        if not os.path.exists(time_dir):
            os.makedirs(time_dir)

        return outputs_dir, time_dir


def create_image_folder_with_current_time_stamp(output_folder, folder_name_date):

    images_folder = os.path.join(output_folder, "images", folder_name_date)

    if os.makedirs(images_folder):
        sys.exit("Error creating output image folder")

    return images_folder, folder_name_date

def create_sub_folder_for_segmentation(output_folder):
    raw_folder = os.path.join(output_folder, "raw")
    overlay_folder = os.path.join(output_folder, "overlay")
    index_folder = os.path.join(output_folder, "index")

    if os.makedirs(raw_folder):
        sys.exit("Error creating raw folder")
    if os.makedirs(overlay_folder):
        sys.exit("Error creating overlay folder")
    if os.makedirs(index_folder):
        sys.exit("Error creatinfg index folder")

    return raw_folder, overlay_folder, index_folder

def create_name_date():
    from datetime import datetime

    date = datetime.now()
    folder_name_date = str(date.year)+ "_" + str(date.month) + "_" + str(date.day) + "_" + str(date.hour) + "_" + str(date.minute)

    return folder_name_date




        

        
