from functools import partial as Partial

import torch
import torch.nn as nn
from torch.ao.quantization import prepare, convert

from . import Configurations
from .Configurations import prepare_custom_config_dict, convert_custom_config_dict

from .Utilities import astype

def propagate(model, what, allow_list = None):
    assert isinstance(model, nn.Module)

    if allow_list is not None:
        assert isinstance(allow_list, list)

    for name, module in model.named_children():
        if allow_list is None or type(module) in allow_list:
            for w in what:
                if hasattr(model, w):
                    assert type(model.__getattribute__(w)) == what[w]

                    module.__setattr__(w, model.__getattribute__(w))

def quantize(float_model, N, Es, nocast = False, device = 'cpu', fpga_host = '127.0.0.1', fpga_port = 8080, fpga_conf = 'KV260 4 4 8 2 128 8', max_workers = 1):
    try:
        qconfig = Configurations.__getattribute__(f'default_posit{N}_{Es}_qconfig')
    except:
        raise RuntimeError(f'Could not find a QConfig for Posit ({N}, {Es}). Allowed QConfig for Posit are (N = 8, Es = 0), (N = 16, Es = 1) or (N = [1, 32], Es = 2).')

    assert device in ['cpu', 'fpga'], f'Expect <device> as CPU or FPGA, is {device}'

    float_model.qconfig = qconfig
    float_model.qdevice = device
    float_model.fpga_host = fpga_host
    float_model.fpga_port = fpga_port
    float_model.fpga_conf = fpga_conf
    float_model.max_workers = max_workers
    float_model.nocast = nocast

    propagate(float_model, {'nocast': bool, 'qdevice': str, 'fpga_host': str, 'fpga_port': int, 'fpga_conf': str , 'max_workers': int})

    quantized_model = convert(prepare(float_model, prepare_custom_config_dict = prepare_custom_config_dict), inplace = True, convert_custom_config_dict = convert_custom_config_dict)

    return quantized_model

def _f(float_model, p_dtype = None):
    if hasattr(float_model, 'weight'):
        device = float_model.weight.data.device
        dtype = float_model.weight.data.dtype

        float_model.weight.data = torch.as_tensor(astype(astype(float_model.weight.data, p_dtype), float), device = device, dtype = dtype)

    if hasattr(float_model, 'bias'):
        device = float_model.bias.data.device
        dtype = float_model.bias.data.dtype

        float_model.bias.data = torch.as_tensor(astype(astype(float_model.bias.data, p_dtype), float), device = device, dtype = dtype)

def quantize_weights(float_model, N = 8, Es = 0):
    try:
        qconfig = Configurations.__getattribute__(f'default_posit{N}_{Es}_qconfig')
    except:
        raise RuntimeError(f'Could not find a QConfig for Posit ({N}, {Es}). Allowed QConfig for Posit are (N = 8, Es = 0), (N = 16, Es = 1) or (N = [1, 32], Es = 2).')

    p_dtype = qconfig.weight().dtype
    
    fn = Partial(_f, p_dtype = p_dtype)

    float_model.apply(fn)
