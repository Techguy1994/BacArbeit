import argparse
import logging
import os
import pyarmnn as ann
import tflite_runtime.interpreter as tflite

def tflite_runtime():
    print("tflite")

def pyarmnn():
    print("pyarmnn")

def openvino():
    print("openvino")

def onnx_runtime():
    print("onnx")

def pytorch():
    print("Pytorch")




def main():
    print("Main")

    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module')

    parser.add_argument("-tfl", '--tflite_model', help='TFLite model file', required=False)
    parser.add_argument("-sd", '--save_dir', default='./tmp', help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=10, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=0, type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-thr", '--threads', default=1, type=int, help='number of threads to run compiled bench on', required=False)
    parser.add_argument("-bf", '--bench_file', default="linux_aarch64_benchmark_model", type=str, help='path to compiled benchmark file', required=False)

    parser.add_argument('--print', dest='print', action='store_true')
    parser.add_argument('--no-print', dest='print', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--interpreter', dest='interpreter', action='store_true')
    parser.add_argument('--no-interpreter', dest='interpreter', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--pyarmnn', dest='pyarmnn', action='store_true')
    parser.add_argument('--no-pyarmnn', dest='pyarmnn', action='store_false')
    parser.set_defaults(feature=False)

    args = parser.parse_args()
    
    if not args.pyarmnn and not args.interpreter and not args.bench_file:
        logging.error("No Runtime chosen, please choose either PyARMNN, the TFLite Interpreter or provide a compiled benchmark file")
        return

    # if TFLite model is provided, use it for inference
    if args.tflite_model and os.path.isfile(args.tflite_model):
        #run_network(tflite_path=args.tflite_model, save_dir=args.save_dir, niter=args.niter,
        #                print_bool=args.print, sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn)
        print("tflite")
    else:
        # if no neural network models are provided, return
        logging.error("Invalid model path {} passed.".format(args.tflite_model))
        return

    # run inference using the provided benchmark file if the benchmark file is valid
    if args.bench_file and os.path.isfile(args.bench_file):
        #run_compiled_bench(tflite_path=args.tflite_model, save_dir=args.save_dir, niter=args.niter, print_bool=args.print, 
        #sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn, bench_file=args.bench_file, num_threads = args.threads)
        print("benchmark")

    logging.info("\n**********RPI INFERENCE DONE**********")


if __name__ == "__main__":
    main()

