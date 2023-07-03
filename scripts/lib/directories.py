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
    print(args)
    return 0


def handle_label_directory(args):
    print(args)
    return 0

def handle_output_directory(args):
    print(args)
    return 0

        
