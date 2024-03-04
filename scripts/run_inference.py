import lib.arguments as ag
import lib.data as dat
import lib.directories as d
import os 
import sys


def main():
    args = ag.Arguments()

    name_date = d.create_name_date()
    #print(args.model.split("/")[-1])

    print(args.model, args.api, args.type)

    #sys.exit()

    if args.type == "class":
        global cl
        import inf.classification as cl
        run_classification(args, name_date)
    if args.type == "det":
        global det
        import inf.detection as det
        run_detection(args, name_date)
    if args.type == "seg":
        global seg
        import inf.segmentation as seg
        run_segmentation(args, name_date)

def run_classification(args, name_date):
    if args.profiler == "cprofiler":
        import cProfile, pstats
        profiler = cProfile.Profile()
        profiler.enable()

    if args.api == "tf" or args.api == "delegate": 
        df = cl.run_tf(args)
    if args.api == "pyarmnn":
        df = cl.run_pyarmnn(args)
    if args.api == "onnx":
        df = cl.run_onnx(args)
    if args.api == "pytorch":
        df = cl.run_pytorch(args)
    if args.api == "ov":
        print(args.a_sync)
        if args.a_sync:
            df = cl.run_async_ov(args)
        else:
            df = cl.run_sync_ov(args)

    print(df)
    dat.store_pandas_data_frame_as_csv(df, args.output, name_date, args.type, args.model, args.api, args.skip_output)

    if args.profiler == "cprofiler":
        profiler.disable()

        with open(os.path.join(args.time_dir, name_date), 'w') as stream:
            stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
            stats.print_stats()

    
def run_detection(args, name_date):
    #print(name_date)

    output_image_folder, name_date = d.create_image_folder_with_current_time_stamp(args.output, name_date)
    #print(name_date)
    #sys.exit()

    if args.profiler == "cprofiler":
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    if args.api == "tf" or args.api == "delegate": 
        df = det.run_tf(args, output_image_folder)
    if args.api == "pyarmnn":
        df = det.run_pyarmnn(args, output_image_folder)
    if args.api == "onnx":
        df = det.run_onnx(args, output_image_folder)
    if args.api == "pytorch":
        df = det.run_pytorch(args, output_image_folder)
    if args.api == "ov":
        df = det.run_sync_ov(args, output_image_folder)

    dat.store_pandas_data_frame_as_csv_det_seg(df, args.output, name_date, args.type, args.model, args.api, args.skip_output)
    
    if args.profiler == "cprofiler":
        profiler.disable()
    
def run_segmentation(args, name_date):
    if args.profiler == "cprofiler":
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    output_image_folder, name_date = d.create_image_folder_with_current_time_stamp(args.output, name_date)
    raw_folder, overlay_folder, index_folder = d.create_sub_folder_for_segmentation(output_image_folder)

    if args.api == "tf" or args.api == "delegate":
        df = seg.run_tf(args, raw_folder, overlay_folder, index_folder)
        
    if args.api == "pyarmnn":
        df = seg.run_pyarmnn(args, raw_folder, overlay_folder, index_folder)
    if args.api == "onnx":
        df = seg.run_onnx(args, raw_folder, overlay_folder, index_folder)
    if args.api == "pytorch":
        df = seg.run_pytorch(args, raw_folder, overlay_folder, index_folder)
    if args.api == "ov":
        df = seg.run_sync_openvino(args, raw_folder, overlay_folder, index_folder)
    
    dat.store_pandas_data_frame_as_csv_det_seg(df, args.output, name_date, args.type, args.model, args.api, args.skip_output)




    if args.profiler == "cprofiler":
        profiler.disable()
    
        
"""
def create_image_folder_with_current_time_stamp(output_folder):
    date = datetime.now()
    folder_name_date = str(date.year)+ "_" + str(date.month) + "_" + str(date.day) + "_" + str(date.hour) + "_" + str(date.minute)

    images_folder = os.path.join(output_folder, "images", folder_name_date)

    val = os.makedirs(images_folder)
    print(val)

    return images_folder, folder_name_date
"""




if __name__ == "__main__":
    main()
