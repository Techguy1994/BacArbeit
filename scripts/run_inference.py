from ast import arg
import numpy as np
import pandas as pd

import lib.arguments as ag
import lib.data as dat
from datetime import datetime
import os


def main():
    args = ag.Arguments()
    if args.type == "class":
        global cl
        import inf.classification as cl
        run_classification(args)
    if args.type == "det":
        global det
        import inf.detection as det
        run_detection(args)
    if args.type == "seg":
        global seg
        import inf.segmentation as seg
        run_segmentation(args)

def run_classification(args):
    if args.profiler == "cprofiler":
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    if args.api == "tf": 
        df = cl.run_tf(args)
        #dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "pyarmnn":
        df = cl.run_pyarmnn(args)
        dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "onnx":
        df = cl.run_onnx(args)
        dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "pytorch":
        print("implement pytorch")
    if args.api == "ov":
        print("implement ov")


    if args.profiler == "cprofiler":
        profiler.disable()

    
        

    
def run_detection(args):

    output_image_folder, name_date = create_image_folder_with_current_time_stamp(args.output)

    if args.profiler == "cprofiler":
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    if args.api == "tf": 
        df = det.run_tf(args)
        #dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "pyarmnn":
        df = det.run_pyarmnn(args)
        #dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "onnx":
        df = det.run_onnx(args, output_image_folder)
        dat.store_pandas_data_frame_as_csv_det_seg(df, args.output, name_date)
    if args.api == "pytorch":
        print("implement pytorch")
    if args.api == "ov":
        print("implement ov")

    
    if args.profiler == "cprofiler":
        profiler.disable()
    
def run_segmentation():
    print("run segmentation")
        

def create_image_folder_with_current_time_stamp(output_folder):
    date = datetime.now()
    folder_name_date = str(date.year)+ "_" + str(date.month) + "_" + str(date.day) + "_" + str(date.hour) + "_" + str(date.minute)

    images_folder = os.path.join(output_folder, "images", folder_name_date)

    val = os.makedirs(images_folder)
    print(val)

    return images_folder, folder_name_date




if __name__ == "__main__":
    main()
