import sys

import argparse
from argparse import ArgumentParser

import numpy as np

import softposit as sp

import torch
import torch.nn as nn

import time

import PQTorch.Configurations as Configurations

from PQTorch.Quantization import quantize

from PQTorch.Utilities import gettype, astype, tobin, frombin

def init_weight_bias(model):
    for name, module in model.named_children():
        if hasattr(module, 'weight'):
            module.weight.data = torch.rand(module.weight.data.shape) * 0.01
        if hasattr(module, 'bias'):
            if module.bias is None:
                return
            module.bias.data = torch.rand(module.bias.data.shape) * 0.01

def main(N = 8, Es = 2, device = 'cpu', fpga_host = '127.0.0.1', fpga_port = 8080, fpga_conf = 'KV260 4 4 8 2 128 8'):
    try:
        qconfig = Configurations.__getattribute__(f'default_posit{N}_{Es}_qconfig')
    except:
        raise RuntimeError(f'Could not find a QConfig for Posit ({N}, {Es}). Allowed QConfig for Posit are (N = 8, Es = 0), (N = 16, Es = 1) or (N = [1, 32], Es = 2).')

    model = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(256, 8, 1, stride = 1, padding = 0),
            )

    init_weight_bias(model)

    model.eval()

    quantized_model_cpu = quantize(model, N, Es)

    if device == 'fpga':
        quantized_model_fpga = quantize(model, N, Es, nocast = True, device = device, fpga_host = fpga_host, fpga_port = fpga_port, fpga_conf = fpga_conf, max_workers = 4)

    input = torch.rand([1, 64, 16, 9]) * 0.01
    qinput = astype(input, qconfig.weight().dtype)

    if device == 'fpga':
        bqinput = tobin(qinput)

    now = time.time()
    output = model(input)
    float_model_inference_time = time.time() - now

    now = time.time()
    qoutput = quantized_model_cpu(qinput)
    quantized_model_cpu_inference_time = time.time() - now

    if device == 'fpga':
        now = time.time()
        bqoutput = quantized_model_fpga(bqinput)
        quantized_model_fpga_inference_time = time.time() - now

    output = astype(output, np.double)
    qoutput = astype(qoutput, np.double)

    if device == 'fpga':
        bqoutput = astype(frombin(bqoutput, qconfig.weight().dtype), np.double)

    err_cpu = np.sqrt(np.mean((qoutput - output)**2)) / np.sqrt(np.mean(output**2)) * 100

    if device == 'fpga':
        err_fpga = np.sqrt(np.mean((bqoutput - output)**2)) / np.sqrt(np.mean(output**2)) * 100

    print(f'Inference Err. on CPU: {err_cpu}%')

    if device == 'fpga':
        print(f'Inference Err. on FPGA: {err_fpga}%')

    print(f'float_model_inference_time: {float_model_inference_time}')
    print(f'quantized_model_cpu_inference_time: {quantized_model_cpu_inference_time}')

    if device == 'fpga':
        print(f'quantized_model_fpga_inference_time: {quantized_model_fpga_inference_time}')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--N', type = int, default = 8)
    parser.add_argument('--Es', type = int, default = 2)
    parser.add_argument('--device', type = str, default = 'fpga')
    parser.add_argument('--fpga_host', type = str, default = '127.0.0.1')
    parser.add_argument('--fpga_port', type = int, default = 8080)
    parser.add_argument('--fpga_conf', type = str, default = 'KV260 4 4 8 2 128 8', help = 'String in the Format \"Part Rows Columns N Es QSize Depth\".')

    args = parser.parse_args()

    main(args.N, args.Es, args.device, args.fpga_host, args.fpga_port, args.fpga_conf)
