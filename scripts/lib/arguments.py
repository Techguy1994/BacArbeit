import argparse
import lib.directories as d
import sys

class Arguments:
    def __init__(self):
        print("Hi")
        self.args = self.arguments()
        self.model = d.handle_model_directory(self.args)
        self.images = d.handle_image_directory(self.args)
        self.label = d.handle_label_directory(self.args)
        self.label = self.create_label_list()
        self.colormap = self.handle_colormap_for_seg()
        self.type = self.handle_type()
        self.api = self.handle_api()
        self.profiler = self.handle_profiler_selection()
        self.output, self.time = d.handle_output_directory(self.args, self.api, self.type, self.model, self.profiler)
        #print(self.output)
        self.niter = int(self.args.niter)
        self.thres = float(self.args.threshold)
        self.sleep = int(self.args.sleep)
        self.n_big = int(self.args.n_big)
        self.a_sync = self.args.a_sync
        self.skip_output = self.args.skip_output
        self.randomized_input = self.args.randomized_input
        #print(int(self.randomized_input))
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
        parser.add_argument("-thres", "--threshold", help="threshold for object detectio", required=False, default=0.25)
        parser.add_argument("-ho", "--handle_output", help="defines wheter the output is handled", required=False, action="store_true", default="store_false")
        parser.add_argument("-p", "--profiler", help="define which profiler is to be used", required=False, default="perfcounter")
        parser.add_argument("-big", "--n_big", help=" n biggest results", required=False, default=5)
        parser.add_argument("-async", "--a_sync", help="in combination with ov inference, handles if it runs in sync or async mode, default is sync", required=False, action="store_true")
        parser.add_argument("-c", "--colormap", help="colormap chosen depending on the data set", required=False)

        parser.add_argument("-so", "--skip_output", help="if provided this skips the output handling", action="store_true")
        parser.add_argument("-ri", "--randomized_input", help="randomizes inputs for the input, give a of the number of the input", required=False)

        return parser.parse_args()
    
    def handle_type(self):
        if self.args.type not in ["class", "det", "seg"]:
        #if self.args.type != "class" and self.args.type != "det" and self.args.type != "seg":
            sys.exit("Error: Type given is not defined or not given all! Please state class, det or seg")
        else: 
            return self.args.type

    def handle_api(self):
        #if self.args.api != "tf" and "pyarmn":
        if self.args.api not in ["tf", "pyarmnn", "ov", "onnx", "pytorch", "delegate"]:
            sys.exit("Error: Api given is not defined or not given all! Please state pyarmnn or tf or ov or onnx or pytorch")
        else: 
            return self.args.api
        
    def handle_profiler_selection(self):
        if self.args.profiler not in ["perfcounter", "cprofiler", "pyarmnn", "pytorch", "onnx"]:
            sys.exit("Error: profiler given not a valid entry please choose between perfcounter, cprofiler, pyarmnn, pytorch and onnx")
        else:
            return self.args.profiler
        
    def create_label_list(self):
        with open(self.label, "r") as f:
            categories = [s.strip() for s in f.readlines()]
    
        return categories
    
    def handle_colormap_for_seg(self):
        if self.args.type == "seg":
            if self.args.colormap == "ade20k":
                #print(args.dataset)
                return create_ade20k_label_colormap()
            elif self.args.colormap == "pascal_voc_2012":
                return create_pascal_label_colormap()
            elif self.args.colormap == "cityscapes":
                return create_cityscapes_label_colormap()
            else:
                sys.exit("No vaild name for dataset given")
        
        return 0

        
def create_pascal_label_colormap_old():
    import numpy as np
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    #print("colormap: --- >", colormap.shape)
    #print(colormap)
    #print("finto")
    return colormap

def create_pascal_label_colormap(N=256, normalized=False):
    import numpy as np

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    print("cmap shape: ", cmap.shape)
    print("cmap: ", cmap)

    return cmap

def create_cityscapes_label_colormap():
    import numpy as np
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap

def create_ade20k_label_colormap():
    import numpy as np
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    """
    return np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])
        
    #def handle_other_parameters(self):
    #    if self.args.niter:
    #        self.niter = self.args.niter

            