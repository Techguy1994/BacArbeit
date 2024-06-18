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
        import io
        profiler = cProfile.Profile()
        profiler.enable()

    if args.api == "tf" or args.api == "delegate": 
        df = cl.run_tf(args)
    if args.api == "pyarmnn":
        df = cl.run_pyarmnn(args)
    if args.api == "onnx":
        if args.profiler == "onnx":
            df, session = cl.run_onnx(args)
        else:
            df = cl.run_onnx(args)
    if args.api == "pytorch":
        if args.profiler == "pytorch":
            df = cl.run_pytorch_with_profiler(args)
        else:
            df, profiler = cl.run_pytorch(args)
    if args.api == "ov":
        df = cl.run_sync_ov(args)

    print(df)
    dat.store_pandas_data_frame_as_csv(df, name_date, args)

    if args.profiler == "cprofiler":
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open('temp.txt', 'w+') as f:
            f.write(s.getvalue())
        
        save_cprofiler_csv(args, name_date)
    elif args.profiler == "onnx":
        prof_file = session.end_profiling()
        print(prof_file)
        print(os.path.join(args.time, prof_file))
        os.rename(prof_file, os.path.join(args.time, prof_file))
    elif args.profiler == "pytorch":
        print(args.time)
        os.rename("temp.json", os.path.join(args.time, name_date + "_pytorch_profiler.json"))


        
        


    
def run_detection(args, name_date):
    #print(name_date)

    output_image_folder, name_date = d.create_image_folder_with_current_time_stamp(args.output, name_date)
    #print(name_date)
    #sys.exit()

    if args.profiler == "cprofiler":
        import cProfile, pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()

    if args.api == "tf" or args.api == "delegate": 
        df = det.run_tf(args, output_image_folder)
    if args.api == "pyarmnn":
        df = det.run_pyarmnn(args, output_image_folder)
    if args.api == "onnx":
        if args.profiler == "onnx":
            df, session = det.run_onnx(args, output_image_folder)
        else:
            df = det.run_onnx(args, output_image_folder)
    if args.api == "pytorch":
        df = det.run_pytorch(args, output_image_folder)
    if args.api == "ov":
        df = det.run_sync_ov(args, output_image_folder)

    #dat.store_pandas_data_frame_as_csv_det_seg(df, args.output, name_date, args.type, args.model, args.api, args.skip_output)
    dat.store_pandas_data_frame_as_csv(df, name_date, args)
    
    if args.profiler == "cprofiler":
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open('temp.txt', 'w+') as f:
            f.write(s.getvalue())
        
        save_cprofiler_csv(args, name_date)
    
    elif args.profiler == "onnx":
        prof_file = session.end_profiling()
        print(prof_file)
        print(os.path.join(args.time, prof_file))
        os.rename(prof_file, os.path.join(args.time, prof_file))
    elif args.profiler == "pytorch":
        print(args.time)
        os.rename("temp.json", args.time)
    
def run_segmentation(args, name_date):
    if args.profiler == "cprofiler":
        import cProfile, pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()

    output_image_folder, name_date = d.create_image_folder_with_current_time_stamp(args.output, name_date)
    raw_folder, overlay_folder, index_folder = d.create_sub_folder_for_segmentation(output_image_folder)

    if args.api == "tf" or args.api == "delegate":
        df = seg.run_tf(args, raw_folder, overlay_folder, index_folder)
        
    if args.api == "pyarmnn":
        df = seg.run_pyarmnn(args, raw_folder, overlay_folder, index_folder)
    if args.api == "onnx":
        if args.prfiler == "onnx":
            df, session = seg.run_onnx(args, raw_folder, overlay_folder, index_folder)
        else: 
            df = seg.run_onnx(args, raw_folder, overlay_folder, index_folder)
    if args.api == "pytorch":
        if args.profiler == "pytroch":
            df = seg.run_pytorch_with_profiler(args, raw_folder, overlay_folder, index_folder)
        else:
            df = seg.run_pytorch(args, raw_folder, overlay_folder, index_folder)
    if args.api == "ov":
        df = seg.run_sync_openvino(args, raw_folder, overlay_folder, index_folder)
    
    dat.store_pandas_data_frame_as_csv(df, name_date, args)




    if args.profiler == "cprofiler":
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open('temp.txt', 'w+') as f:
            f.write(s.getvalue())
        
        save_cprofiler_csv(args, name_date)
    
    elif args.profiler == "onnx":
        prof_file = session.end_profiling()
        print(prof_file)
        print(os.path.join(args.time, prof_file))
        os.rename(prof_file, os.path.join(args.time, prof_file))
    elif args.profiler == "pytorch":
        print(args.time)
        os.rename("temp.json", args.time)
    
        

def save_cprofiler_csv(args, name_date):
    import pandas as pd

    with open('temp.txt') as f:
        lines = f.readlines()



        ncalls_list = []
        tottime_list = []
        percall_tot_list = []
        cumtime_list = []
        percall_cum_list = []
        filename_list = []

        start = 0
        for line in lines:
            if start < 5:
                start = start + 1
            else:
                print(line)

                ncalls = line[0:9]
                tottime = line[10:18]
                percall_tot = line[19:27]
                cum_time = line[28:36]
                percall_cum = line[37:45]
                filename = line[46:-1]

                ncalls_list.append(ncalls)
                tottime_list.append(tottime)
                percall_tot_list.append(percall_tot)
                cumtime_list.append(cum_time)
                percall_cum_list.append(percall_cum)
                filename_list.append(filename)
        
        profiler_dict = {
             "ncalls": ncalls_list,
             "tottime": tottime_list,
             "percall_tot": percall_tot_list,
             "cumtime": cumtime_list,
             "percall_cum": percall_cum_list,
             "filename:lineno(function)": filename_list

        }

        #print(profiler_dict)

        df = pd.DataFrame(profiler_dict)

        print(df.head(5))
        cprofiler_name = os.path.join(args.time, name_date + ".csv")
        df.to_csv(cprofiler_name)
        print(cprofiler_name)



if __name__ == "__main__":
    main()
