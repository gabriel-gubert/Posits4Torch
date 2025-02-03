# Posits4Torch
Field-Programmable Gate Array (FPGA)-enabled Posit Quantization Customization for PyTorch
## Installation
```
git clone --recursive https://github.com/gabriel-gubert/Posits4Torch.git
cd Posits4Torch
source install.sh
```
## Getting Started
The following is a step-by-step guide on how to get started with the main functionality of Posits4Torch.
1. Define and initialize the model.
```
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(64, 256, 3, stride = 1, padding = 1),
    nn.ReLU(),
    nn.Conv2d(256, 256, 1, stride = 1, padding = 0),
    nn.ReLU(),
    nn.Conv2d(256, 256, 1, stride = 1, padding = 0),
    nn.ReLU(),
    nn.Conv2d(256, 8, 1, stride = 1, padding = 0),
)

for name, module in model.named_children():
        if hasattr(module, 'weight'):
            module.weight.data = 0.01 * torch.rand(module.weight.data.shape)
        if hasattr(module, 'bias'):
            if module.bias is None:
                return
            module.bias.data = 0.01 * torch.rand(module.bias.data.shape)

model.eval()
```
2. Configure the Posit quantization. Should the model be run on the Field-Programmable Gate Array (FPGA), one must define the Posit precision (N), exponent bit-length (Es), the device which to deploy the accelerator (Part), number of rows (R) and columns (C) in the 2D Posit MAC Unit Array, its quire size (QSize), and the corresponding depth of the First-In First-Out (FIFO) buffers. Currently supported configurations for the Posit quantization are described in [Table 1](#table-1-supported-posit-quantization-configurations), and in [Table 2](#table-2-supported-hardware-accelerator-configurations) for the hardware accelerator.
```
N = 8           # Posit Precision
Es = 2          # Exponent Bit-length
Part = 'KV260'  # Device
R = 4           # No. of Rows in the 2D RxC Posit MAC Unit Array
C = 4           # No. of Columns in the 2D RxC Posit MAC Unit Array
QSize = 128     # Size of the Posit Quire
Depth = 8       # First-In First-Out (FIFO) Buffer Length
```
3. Quantize the model using `Posits4Torch.Quantization.quantize`.
```
from Posits4Torch.Quantization import quantize

quantized_model_fpga = quantize(
    model, 
    N, 
    Es, 
    nocast = True, 
    device = 'fpga', 
    fpga_host = '127.0.0.1', 
    fpga_port = 8080, 
    fpga_conf = f'{Part} {R} {C} {N} {Es} {QSize} {Depth}',
    max_workers = 4
)
```
4. Run inference with the model. Should the model be run on the FPGA, one must use `Posits4Torch.Utilities.tobin` to feed the model with the input Posit tensor, and `Posits4Torch.Utilities.frombin` to correctly interpret the output tensor back to Posit, which can then be cast back to floating-point using `Posits4Torch.Utilities.astype`.
```
from Posits4Torch.Utilities import gettype, astype, tobin, frombin
import numpy as np
import time

input = 0.01 * torch.rand([1, 64, 16, 9])
input = tobin(astype(input, gettype(N, Es)))

now = time.time()
output = quantized_model_fpga(input)
inference_time = time.time() - now

output = astype(frombin(output, gettype(N, Es)), np.double)

print(f'Total Inference Time: {inference_time} s')
```
## Supported Configurations
### Table 1: Supported Posit Quantization Configurations.
| Posit Precision (N)  | Exponent Bit-length (Es) |
|----|----|
| 1  | 2  |
| 2  | 2  |
| 3  | 2  |
| 4  | 2  |
| 5  | 2  |
| 6  | 2  |
| 7  | 2  |
| 8  | 2  |
| 9  | 2  |
| 10 | 2  |
| 11 | 2  |
| 12 | 2  |
| 13 | 2  |
| 14 | 2  |
| 15 | 2  |
| 16 | 2  |
| 17 | 2  |
| 18 | 2  |
| 19 | 2  |
| 20 | 2  |
| 21 | 2  |
| 22 | 2  |
| 23 | 2  |
| 24 | 2  |
| 25 | 2  |
| 26 | 2  |
| 27 | 2  |
| 28 | 2  |
| 29 | 2  |
| 30 | 2  |
| 31 | 2  |
### Table 2: Supported Hardware Accelerator Configurations.
| Part Name (Part)  | Rows (R) | Columns (C) | Posit Precision (N) | Exponent Bit-length (Es) | Quire Size (QSize) | FIFO Depth (Depth) |
|-------|---|---|---|----|-------|-------|
| KV260 | 8 | 8 | 8 | 2  | 128   | 256   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 128   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 64    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 16    |
| KV260 | 8 | 8 | 6 | 2  | 128   | 512   |
| KV260 | 4 | 4 | 8 | 2  | 128   | 32    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 256   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 16    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 8     |
| KV260 | 8 | 8 | 8 | 2  | 128   | 8     |
| KV260 | 8 | 8 | 7 | 2  | 128   | 512   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 512   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 32    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 128   |
| KV260 | 4 | 4 | 8 | 2  | 128   | 64    |