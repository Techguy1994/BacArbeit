import argparse
import lib.directories as d
import sys


class Arguments:
    def __init__(self):
        self.args = self.arguments()
        self.model = d.handle_model_directory(self.args)
        self.images = d.handle_image_directory(self.args)
        self.label = d.handle_label_directory(self.args)
        self.label = self.create_label_list()
        self.type = self.handle_type()
        self.api = self.handle_api()
        self.profiler = self.handle_profiler_selection()
        self.output = d.handle_output_directory(self.args, self.api, self.type, self.model, self.profiler)
        print(self.output)
        self.niter = int(self.args.niter)
        self.thres = float(self.args.threshold)
        self.sleep = int(self.args.sleep)
        self.n_big = int(self.args.n_big)
        self.a_sync = self.args.a_sync
        #handle_other_paramters()

    def arguments(self):
        parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

        #basic settings
        parser.add_argument("-api", '--api', help='inference API: pyarmnn, tf, ov, onnx, pytorch', required=True) #which api to use
        parser.add_argument("-t", '--type', help='which type of inference, class, det or seg', required=True) #which inference type chosen

        #directories
        #model directory 
        parser.add_argument("-m", '--model', help='model name', required=False)
        parser.add_argument("-mp", '--model_path', help='model path', required=False)

        #image directories
        parser.add_argument("-img", "--image", help="path of a picture", required=False)
        parser.add_argument("-imgp", "--image_folder", help="image folder path", required=False)
        parser.add_argument("-imgd", "--image_folder_default", help="use default image folder path", required=False, action="store_true")

        #label
        parser.add_argument("-l", "--labels", help="txt file with classes", required=False)
        parser.add_argument("-lp", "--labels_path", help="path of the txt file with classes", required=False)

        #result folder
        parser.add_argument("-op", "--output", help="name of the output folder ", required=False)
        parser.add_argument("-opp", "--output_path", help="where the results are saved specify concrete output path to folder", required=False)
        parser.add_argument("-opd", "--output_default", help="default mechanism where the results will be saved", required=False, action="store_true")

        #other arguments 
        parser.add_argument("-s", "--sleep", help="when number of iterations defines time gap", required=False, default=0)
        parser.add_argument("-ni", "--niter", help="number of iterations", required=False, default=1)
        parser.add_argument("-thres", "--threshold", help="threshold for object detectio", required=False, default=0.5)
        parser.add_argument("-ho", "--handle_output", help="defines wheter the output is handled", required=False, action="store_true", default="store_false")
        #parser.add_argument("rand", "--randomized_input", help="specifgfy if the input image should be randomized", required=False, action="store_true")
        parser.add_argument("-p", "--profiler", help="define which profiler is to be used", required=False, default="perfcounter")
        parser.add_argument("-big", "--n_big", help=" n biggest results", required=False, default=3)
        parser.add_argument("-async", "--a_sync", help="in combination with ov inference, handles if it runs in sync or async mode, default is sync", required=False, action="store_true")


        return parser.parse_args()
    
    def handle_type(self):
        if self.args.type not in ["class", "det", "seg"]:
        #if self.args.type != "class" and self.args.type != "det" and self.args.type != "seg":
            sys.exit("Error: Type given is not defined or not given all! Please state class, det or seg")
        else: 
            return self.args.type

    def handle_api(self):
        #if self.args.api != "tf" and "pyarmn":
        if self.args.api not in ["tf", "pyarmnn", "ov", "onnx", "pytorch"]:
            sys.exit("Error: Api given is not defined or not given all! Please state pyarmnn or tf or ov or onnx or pytorch")
        else: 
            return self.args.api
        
    def handle_profiler_selection(self):
        if self.args.profiler not in ["perfcounter", "cprofiler", "pyarmnn", "pytorch", "onnx"]:
            sys.exit("Error: profiler given not a valid entry please choose between perfcounter, cprofiler, pyarmnn, pytorch and onnx")
        else:
            return self.args.profiler
        
    def create_label_list(self):
        if args.type == "seg":
            if args.colormap == "ade20k":
                #print(args.dataset)
                colormap = create_ade20k_label_colormap()
            elif args.colormap == "pascal_voc_2012":
                colormap = create_pascal_label_colormap()
            elif args.colormap == "cityscapes":
                colormap = create_cityscapes_label_colormap()
            else:
                sys.exit("No vaild name for dataset given")
        else:
            with open(self.label, "r") as f:
                categories = [s.strip() for s in f.readlines()]
        
            return categories
        
    #def handle_other_parameters(self):
    #    if self.args.niter:
    #        self.niter = self.args.niter

            