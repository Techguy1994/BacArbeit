import lib.postprocess as post
import lib.data as dat

def output_onnx_class(result, output_data_type, label, n_big, output_dict, image, lat):
    output = post.handle_output_onnx_mobilenet_class(result, output_data_type, label, n_big)
    output_dict = dat.store_output_dictionary_class(output_dict, image, lat, output, n_big)
    return output_dict