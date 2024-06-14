from . import *

from .Configurations import convert_custom_config_dict, prepare_custom_config_dict
from .Configurations import default_posit8_0_qconfig
from .Configurations import default_posit16_1_qconfig
from .Configurations import default_posit32_2_qconfig
from .Configurations import default_posit1_2_qconfig
from .Configurations import default_posit2_2_qconfig
from .Configurations import default_posit3_2_qconfig
from .Configurations import default_posit4_2_qconfig
from .Configurations import default_posit5_2_qconfig
from .Configurations import default_posit6_2_qconfig
from .Configurations import default_posit7_2_qconfig
from .Configurations import default_posit8_2_qconfig
from .Configurations import default_posit9_2_qconfig
from .Configurations import default_posit10_2_qconfig
from .Configurations import default_posit11_2_qconfig
from .Configurations import default_posit12_2_qconfig
from .Configurations import default_posit13_2_qconfig
from .Configurations import default_posit14_2_qconfig
from .Configurations import default_posit15_2_qconfig
from .Configurations import default_posit16_2_qconfig
from .Configurations import default_posit17_2_qconfig
from .Configurations import default_posit18_2_qconfig
from .Configurations import default_posit19_2_qconfig
from .Configurations import default_posit20_2_qconfig
from .Configurations import default_posit21_2_qconfig
from .Configurations import default_posit22_2_qconfig
from .Configurations import default_posit23_2_qconfig
from .Configurations import default_posit24_2_qconfig
from .Configurations import default_posit25_2_qconfig
from .Configurations import default_posit26_2_qconfig
from .Configurations import default_posit27_2_qconfig
from .Configurations import default_posit28_2_qconfig
from .Configurations import default_posit29_2_qconfig
from .Configurations import default_posit30_2_qconfig
from .Configurations import default_posit31_2_qconfig

from .Observers import LazyObserver
from .Quantization import propagate, quantize, quantize_weights
from .Quantized import BatchNorm2d, Conv2d, MaxPool2d, ReLU
from .Utilities import gettype, astype, tobin, frombin
