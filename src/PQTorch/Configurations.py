import torch
import torch.nn as nn
from torch.ao.quantization import QConfig

import softposit as sp

from .Observers import LazyObserver

from . import Quantized

prepare_custom_config_dict = {
    'float_to_observed_custom_module_class': {
        nn.BatchNorm2d: Quantized.BatchNorm2d,
        nn.Conv2d: Quantized.Conv2d,
        nn.MaxPool2d: Quantized.MaxPool2d,
        nn.ReLU: Quantized.ReLU
    }
}

convert_custom_config_dict = {
    'observed_to_quantized_custom_module_class': {
        Quantized.BatchNorm2d: Quantized.BatchNorm2d,
        Quantized.Conv2d: Quantized.Conv2d,
        Quantized.MaxPool2d: Quantized.MaxPool2d,
        Quantized.ReLU: Quantized.ReLU
    }
}

default_posit8_0_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit8), weight = LazyObserver.with_args(dtype = sp.posit8))
default_posit16_1_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit16), weight = LazyObserver.with_args(dtype = sp.posit16))
default_posit32_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit32), weight = LazyObserver.with_args(dtype = sp.posit32))
default_posit1_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit1_2), weight = LazyObserver.with_args(dtype = sp.posit1_2))
default_posit2_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit2_2), weight = LazyObserver.with_args(dtype = sp.posit2_2))
default_posit3_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit3_2), weight = LazyObserver.with_args(dtype = sp.posit3_2))
default_posit4_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit4_2), weight = LazyObserver.with_args(dtype = sp.posit4_2))
default_posit5_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit5_2), weight = LazyObserver.with_args(dtype = sp.posit5_2))
default_posit6_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit6_2), weight = LazyObserver.with_args(dtype = sp.posit6_2))
default_posit7_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit7_2), weight = LazyObserver.with_args(dtype = sp.posit7_2))
default_posit8_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit8_2), weight = LazyObserver.with_args(dtype = sp.posit8_2))
default_posit9_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit9_2), weight = LazyObserver.with_args(dtype = sp.posit9_2))
default_posit10_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit10_2), weight = LazyObserver.with_args(dtype = sp.posit10_2))
default_posit11_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit11_2), weight = LazyObserver.with_args(dtype = sp.posit11_2))
default_posit12_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit12_2), weight = LazyObserver.with_args(dtype = sp.posit12_2))
default_posit13_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit13_2), weight = LazyObserver.with_args(dtype = sp.posit13_2))
default_posit14_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit14_2), weight = LazyObserver.with_args(dtype = sp.posit14_2))
default_posit15_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit15_2), weight = LazyObserver.with_args(dtype = sp.posit15_2))
default_posit16_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit16_2), weight = LazyObserver.with_args(dtype = sp.posit16_2))
default_posit17_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit17_2), weight = LazyObserver.with_args(dtype = sp.posit17_2))
default_posit18_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit18_2), weight = LazyObserver.with_args(dtype = sp.posit18_2))
default_posit19_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit19_2), weight = LazyObserver.with_args(dtype = sp.posit19_2))
default_posit20_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit20_2), weight = LazyObserver.with_args(dtype = sp.posit20_2))
default_posit21_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit21_2), weight = LazyObserver.with_args(dtype = sp.posit21_2))
default_posit22_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit22_2), weight = LazyObserver.with_args(dtype = sp.posit22_2))
default_posit23_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit23_2), weight = LazyObserver.with_args(dtype = sp.posit23_2))
default_posit24_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit24_2), weight = LazyObserver.with_args(dtype = sp.posit24_2))
default_posit25_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit25_2), weight = LazyObserver.with_args(dtype = sp.posit25_2))
default_posit26_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit26_2), weight = LazyObserver.with_args(dtype = sp.posit26_2))
default_posit27_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit27_2), weight = LazyObserver.with_args(dtype = sp.posit27_2))
default_posit28_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit28_2), weight = LazyObserver.with_args(dtype = sp.posit28_2))
default_posit29_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit29_2), weight = LazyObserver.with_args(dtype = sp.posit29_2))
default_posit30_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit30_2), weight = LazyObserver.with_args(dtype = sp.posit30_2))
default_posit31_2_qconfig = QConfig(activation = LazyObserver.with_args(dtype = sp.posit31_2), weight = LazyObserver.with_args(dtype = sp.posit31_2))
