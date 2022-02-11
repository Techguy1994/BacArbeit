import argparse, os, sys
from time import sleep, time
import numpy as np
import tensorflow as tf


def main(args):
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print information about model inputs and outputs
    print("\n*****Model Inputs*****:")
    for input in input_details:
        for v,k in input.items():
            print(v,k)
        print()
    print()

    print("\n*****Model Outputs*****")
    for output in output_details:
        for v, k in output.items():
            print(v,k)
        print()
    print()

    # Test the model on random input data.
    print("\n*****Sleep between inferences is set to: {}s *****".format(args.sleep))
    for i in range(args.niter):
        # print("\n*****Initializing model inputs with random data*****")
        # assume model with only one input
        input_data = np.random.randint(low=np.iinfo(input['dtype']).min, high=np.iinfo(input['dtype']).max,
                                       size=input_details[0]['shape'], dtype=input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]['index'], input_data)

        beg = time()
        interpreter.invoke()
        print("{} inference took: {:.3f}ms".format(i, 1000*(time() - beg)))
        if args.sleep:
            sleep(args.sleep)

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        #args.print
        if True:
            for output in output_details:
                # works only for classification networks so far
                output_data = interpreter.get_tensor(output['index'])
                # find indizes of the 10 highest values in output array
                max_positions = np.argpartition(output_data[0], -10)[-10:]
                out_normalization_factor = np.iinfo(output['dtype']).max # maximal value of output dtype (uint8 -> 255)
                for entry in max_positions:
                    # go over entries and print their position and result in percent
                    val = output_data[0][entry] / out_normalization_factor
                    print("\tpos {} : {:.2f}%".format(entry, val*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TFLite inference benchmark')
    parser.add_argument("-m", '--model_path', help='one tfltite model', required=False)
    parser.add_argument("-n", '--niter', default=10, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-rd", '--report_dir', default='reports', help='Directory to save reports into', required=False)
    parser.add_argument('--print', dest='print', action='store_true')
    parser.add_argument('--no-print', dest='print', action='store_false')
    args = parser.parse_args()

    if args.sleep:
        if args.sleep < 0:
            sys.exit("Sleep argument cannot be negative!")
        elif args.sleep > 10:
            print("\nSleep between inference iterations was set to: {}s !!!\n".format(args.sleep))

    if not os.path.isfile(args.model_path):
        sys.exit("{} is not a valid file!!!".format(args.model_path))

    main(args)

