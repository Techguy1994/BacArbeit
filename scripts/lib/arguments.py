import argparse
import lib.directories as d


class Arguments:
    def __init__(self):
        print("in class")
        self.args = self.handle_arguments()
        d.test()
        print(self.args)

    def handle_arguments(self):
        parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module for Classification')

        #basic settings
        parser.add_argument("-api", '--api', help='inference API', required=True) #which api to use
        parser.add_argument("-t", '--type', help='which type of inference', required=True) #which inference type chosen

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

        #result folder
        parser.add_argument("-op", "--output", help="where the results are saved", required=False)


        
        print("Test worked")
        return parser.parse_args()