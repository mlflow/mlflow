try:
    import numpy as np
    __import_np = True
except ImportError:
    __import_np = False

try:
    import pyspark.ml.param
    __import_pyspark_ml_param = True
except ImportError:
    __import_pyspark_ml_param = False


try:
    import tensorflow as tf
    __import_np = True
except ImportError:
    __import_tf = False

try:
    import torch
    __import_torch = True
except ImportError:
    __import_torch = False

def convert_to_str_if_possible(x) -> str:
    if type(x) == str:
        return x
    
    converter_fns_to_try = [convert_to_str_if_pyspark_mlparam, convert_to_str_if_ndarray, convert_to_str_if_tensorflow_tensor, convert_to_str_if_torch_tensor]

    for converter_fn in converter_fns_to_try:
        possible_str = converter_fn(x)
        if type(possible_str) == str:
            return possible_str

    return str(x)

def convert_to_float_if_possible(x) -> float:
    if type(x) == float:
        return x
    
    converter_fns_to_try = [convert_to_float_if_ndarray, convert_to_float_if_tensorflow_tensor, convert_to_float_if_torch_tensor]

    for converter_fn in converter_fns_to_try:
        possible_float = converter_fn(x)
        if type(possible_float) == float:
            return possible_float

    return float(x)

def convert_to_str_if_pyspark_mlparam(x):
    if __import_pyspark_ml_param == False:
        return x

    if isinstance(x, pyspark.ml.param.Param):
        return x.name

    return x


def convert_to_str_if_ndarray(x):
    if __import_np == False:
        return x

    if isinstance(x, np.ndarray):
        return str(x.tolist())

    return x

def convert_to_float_if_ndarray(x):
    if __import_np == False:
        return x

    if isinstance(x, np.ndarray):
        return float(x.item())

    return x

def convert_to_float_if_torch_tensor(x):
    if __import_torch == False:
        return x

    if isinstance(x, torch.Tensor):
        extracted_tensor_val = x.detach().cpu().item()
        return float(extracted_tensor_val)

    return x

def convert_to_str_if_torch_tensor(x):
    if __import_torch == False:
        return x

    if isinstance(x, torch.Tensor):
        extracted_ndarray = x.detach().cpu().numpy()
        return convert_to_float_if_ndarray(extracted_ndarray)

    return x

def convert_to_float_if_tensorflow_tensor(x):
    if __import_tf == False:
        return x

    if isinstance(x, tf.Tensor):
        return float(x)

    return x

def convert_to_str_if_tensorflow_tensor(x):
    if __import_tf == False:
        return x

    if isinstance(x, tf.Tensor):
        extracted_ndarray = x.numpy()
        return convert_to_str_if_ndarray(extracted_ndarray)

    return x