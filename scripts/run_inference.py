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
        dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "pyarmnn":
        df = cl.run_pyarmnn(args)
        dat.store_pandas_data_frame_as_csv(df, args.output)
    if args.api == "onnx":
        df = cl.run_onnx(args)
        dat.store_pandas_data_frame_as_csv(df, args.output)


    if args.profiler == "cprofiler":
        profiler.disable()

    
        

    
def run_detection():
    print("run_detection")
    
def run_segmentation():
    print("run segmentation")
        

if __name__ == "__main__":
    main()
