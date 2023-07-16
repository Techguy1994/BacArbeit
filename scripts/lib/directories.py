import os 
import sys

def search_file(file_name, directory):

    for dirpath, dirnames, files in os.walk(directory):

        for file in files:
            if file == file_name:
                return os.path.join(dirpath, file_name)

    sys.exit(f"Error: The file {file_name} does not exist in the directory {directory}!")


def handle_model_directory(args):
    
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
    print("Start of image handling")
    #print(args)
    image_list = []

    if args.image_folder: 
        if os.path.exists(args.image_folder):
            for img in os.listdir(args.image_folder):
                if ".jpg" in img and "._" not in img:
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
                if ".jpg" in img and "._" not in img:
                    image_list.append(os.path.join(args.image_folder, img))   
    else:
        sys.exit("Error: No image option chosen as input. You can either give the image with -img , image folder path with -imgp or choose the default image folder with -imgd!")
    
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
        if os.path.exists(args.label_path):
            return args.labels_path
        else: 
            sys.exit("The label file name given does not exit! Please enter a vaild path or give the path name")
    else: 
         sys.exit("Error: No label name or path given. Please Enter with -l a model name or with -lp the model path")

def handle_output_directory(args):
    if args.output:
        return args.output
    elif args.output_default:
        return 1

        
