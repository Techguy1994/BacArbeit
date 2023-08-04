
def handle_output_tf(output_data, output_details, label, n_big):
    import numpy as np
    results = []

    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]
    #print(output_details[0]["dtype"])

    if output_details[0]['dtype'] == np.uint8:
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif output_details[0]['dtype'] == np.float32:
        out_normalization_factor = 1

    

    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        results.append({"label":  label[entry], "index": entry, "value": val})

    return results

def handle_output_pyarmnn(output_data, label, n_big):
    import numpy as np
    results = []

    output_data = output_data[0]
    max_positions = np.argpartition(output_data[0], -n_big)[-n_big:]

    if output_data.dtype == "uint8":
        out_normalization_factor = np.iinfo(output_data.dtype).max
    elif output_data.dtype == "float32":
        out_normalization_factor = 1
    
    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[0][entry] / out_normalization_factor
        results.append({"label": label[entry], "index": entry, "value": val})
        
    return results

def handle_output_onnx(output_data, output_details, label, n_big):
    import numpy as np
    results = []

    output_data = output_data.flatten()
    output_data = softmax(output_data) # this is optional

    max_positions = np.argpartition(output_data, -n_big)[-n_big:]
    out_normalization_factor = 1

    #print(output_details[0]["dtype"])

    if "integer" in output_details:
        print("int")
        quit("not adapted to onnx, please change following code when quantized model is given")
        out_normalization_factor = np.iinfo(output_details[0]['dtype']).max
    elif "float" in output_details:
        #print("float")
        out_normalization_factor = 1


    for entry in max_positions:
        # go over entries and print their position and result in percent
        val = output_data[entry] / out_normalization_factor
        results.append({"label":  label[entry], "index": entry, "value": val})
        
    return results

def softmax(x):
    import numpy as np
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()