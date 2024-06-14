import numpy as np

import softposit as sp

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, ObserverBase

from DeepPeNSieve.layers import *

import typing
from typing import Optional

from .RemoteAccel import RemoteAccel
from .Utilities import POSIT_TO_N_ES_MAPPING, POSIT_TO_UNSIGNED_MAPPING, astype, tobin, frombin

class BatchNorm2d(nn.Module):
    _FLOAT_MODEL = nn.BatchNorm2d

    def __init__(
            self,
            num_features, eps = 1e-05, momentum = 0.1,
            device = None, dtype = None,
            fpga_host = '127.0.0.1', fpga_port = 8080, fpga_conf = 'KV260 4 4 8 2 128 8',
            nocast = False,
            max_workers = 1):
        super(BatchNorm2d, self).__init__()
        
        self.device = device
        self.dtype = dtype

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.fpga_host = fpga_host
        self.fpga_port = fpga_port
        self.fpga_conf = fpga_conf
        self.nocast = nocast
        self.max_workers = max_workers

        Part = str(fpga_conf.split(' ')[0])
        R = int(fpga_conf.split(' ')[1])
        C = int(fpga_conf.split(' ')[2])
        N = int(fpga_conf.split(' ')[3])
        Es = int(fpga_conf.split(' ')[4])
        QSize = int(fpga_conf.split(' ')[5])
        Depth = int(fpga_conf.split(' ')[6])

        if device == 'fpga':
            assert N == POSIT_TO_N_ES_MAPPING[dtype]['N']
            assert Es == POSIT_TO_N_ES_MAPPING[dtype]['Es']

        self.Part = Part
        self.R = R
        self.C = C
        self.N = N
        self.Es = Es
        self.QSize = QSize
        self.Depth = Depth

        self.scale = 1.0
        self.zero_point = 0

        gamma = astype(np.ones(num_features), dtype = dtype)
        beta = astype(np.zeros(num_features), dtype = dtype)
        running_mean = astype(np.zeros(num_features), dtype = dtype)
        running_var = astype(np.ones(num_features), dtype = dtype)

        self.BatchNormalization = BatchNormalization(gamma = gamma, beta = beta, momentum = momentum, running_mean = running_mean, running_var = running_var)


    def forward(self, X):
        torch_Tensor = False
        device = None

        if isinstance(X, torch.Tensor):
            _X = X.numpy(force = True)
            torch_Tensor = True
            device = X.device.type
        elif isinstance(X, np.ndarray):
            _X = X
        else:
            raise TypeError(f'Expect <X> as [<torch.Tensor> | <np.ndarray>], is {type(X)}.')

        if type(_X.take(0)) != self.dtype:
            if not self.nocast:
                _X = astype(_X, self.dtype)

        assert X.ndim == 4, f'Expect <X.shape> as (N, C, H, W), is {X.shape}.'

        N, C, H, W = _X.shape

        assert C == self.BatchNormalization.gamma.size, f'Expect <X.shape[1]> as {self.BatchNormalization.gamma.size} (# of Channel(s)), is {C}.'

        gamma = self.BatchNormalization.gamma
        beta = self.BatchNormalization.beta
        running_mean = self.BatchNormalization.running_mean
        running_var = self.BatchNormalization.running_var

        self.BatchNormalization.gamma = self.BatchNormalization.gamma.repeat(H * W)
        self.BatchNormalization.beta = self.BatchNormalization.beta.repeat(H * W)
        self.BatchNormalization.running_mean = self.BatchNormalization.running_mean.repeat(H * W)
        self.BatchNormalization.running_var = self.BatchNormalization.running_var.repeat(H * W)

        _Y = self.BatchNormalization.forward(_X, train_flg = False)

        self.BatchNormalization.gamma = gamma
        self.BatchNormalization.beta = beta
        self.BatchNormalization.running_mean = running_mean

        self.BatchNormalization.running_var = running_var

        if torch_Tensor:
            if issubclass(type(_Y.take(0)), (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)):
                _Y = astype(_Y, np.double)

            Y = torch.from_numpy(_Y).to(device)
        else:
            Y = _Y

        return Y


    @classmethod
    def from_float(cls, float_model):
        assert type(float_model) == cls._FLOAT_MODEL, f'Expect <float_model> as {_FLOAT_MODEL}, is {type(float_model)}.'
        assert hasattr(float_model, 'qconfig') and isinstance(float_model.qconfig, QConfig), f'Expect <float_model.qconfig> as {QConfig}.'

        weight_post_process = float_model.qconfig.weight()

        assert isinstance(weight_post_process, ObserverBase), f'Expect <weight_post_process> as <ObserverBase>, is {type(weight_post_process)}.'

        assert issubclass(weight_post_process.dtype, (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)), f'Expect <weight_post_process.dtype> as [softposit.posit8 | softposit.posit16 | softposit.posit32 | softposit.posit_2], is {weight_post_process.dtype}.'

        weight_post_process(float_model.weight)

        activation_post_process = float_model.activation_post_process if hasattr(float_model, 'activation_post_process') else float_model.qconfig.activation()

        scale, zero_point = activation_post_process.calculate_qparams()

        max_workers = 1
        if hasattr(float_model, 'max_workers') and type(float_model.max_workers) == int:
            max_workers = float_model.max_workers

        qdevice = 'cpu'
        if hasattr(float_model, 'qdevice') and float_model.qdevice in ['cpu', 'fpga']:
            qdevice = float_model.qdevice

        fpga_host = '127.0.0.1'
        if hasattr(float_model, 'fpga_host') and type(float_model.fpga_host) == str:
            fpga_host = float_model.fpga_host

        fpga_port = 8080
        if hasattr(float_model, 'fpga_port') and type(float_model.fpga_port) == int:
            fpga_port = float_model.fpga_port

        fpga_conf = 'KV260 4 4 8 2 128 8'
        if hasattr(float_model, 'fpga_conf') and type(float_model.fpga_conf) == str:
            fpga_conf = float_model.fpga_conf

        nocast = False
        if hasattr(float_model, 'nocast') and type(float_model.nocast) == bool:
            nocast = float_model.nocast

        quantized_model = cls(
                float_model.num_features, float_model.eps, float_model.momentum,
                device = qdevice, dtype = weight_post_process.dtype,
                fpga_host = fpga_host, fpga_port = fpga_port, fpga_conf = fpga_conf,
                nocast = nocast,
                max_workers = max_workers)

        gamma = astype(float_model.weight.float(), weight_post_process.dtype)
        beta = astype(float_model.bias.float(), weight_post_process.dtype)
        running_mean = astype(float_model.running_mean, weight_post_process.dtype)
        running_var = astype(float_model.running_var, weight_post_process.dtype)

        quantized_model.BatchNormalization.gamma = gamma
        quantized_model.BatchNormalization.beta = beta
        quantized_model.BatchNormalization.running_mean = running_mean
        quantized_model.BatchNormalization.running_var = running_var

        quantized_model.scale = scale
        quantized_model.zero_point = zero_point

        return quantized_model


    @classmethod
    def from_observed(cls, observed_model):
        return observed_model


class Conv2d(nn.Module):
    _FLOAT_MODEL = nn.Conv2d

    def __init__(
            self,
            in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros',
            device = None, dtype = None,
            fpga_host = '127.0.0.1', fpga_port = 8080, fpga_conf = 'KV260 4 4 8 2 128 8',
            nocast = False,
            max_workers = 1):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.device = device
        self.dtype = dtype

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.fpga_host = fpga_host
        self.fpga_port = fpga_port
        self.fpga_conf = fpga_conf
        self.nocast = nocast
        self.max_workers = max_workers

        Part = str(fpga_conf.split(' ')[0])
        R = int(fpga_conf.split(' ')[1])
        C = int(fpga_conf.split(' ')[2])
        N = int(fpga_conf.split(' ')[3])
        Es = int(fpga_conf.split(' ')[4])
        QSize = int(fpga_conf.split(' ')[5])
        Depth = int(fpga_conf.split(' ')[6])

        if device == 'fpga':
            assert N == POSIT_TO_N_ES_MAPPING[dtype]['N']
            assert Es == POSIT_TO_N_ES_MAPPING[dtype]['Es']

            self.default_remoteaccel = RemoteAccel(self.fpga_host, self.fpga_port, -1, Part, R, C, N, Es, QSize, Depth, max_workers)

        self.Part = Part
        self.R = R
        self.C = C
        self.N = N
        self.Es = Es
        self.QSize = QSize
        self.Depth = Depth

        self.weight = astype(np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]), dtype = dtype)
        self.bias = astype(np.zeros(out_channels), dtype = dtype)

        self.Convolution = Convolution(W = self.weight, b = self.bias, stride = stride[0], pad = padding[0])


    def set_weight_bias(self, weight, bias = None):
        assert isinstance(weight, np.ndarray), f'Expect <weight> as <np.ndarray>, is {type(weight)}.'

        _t_w = type(weight.take(0))
        assert _t_w == self.dtype, f'Expect <weight.dtype> as {self.dtype}, is {_t_w}.'

        self.weight = weight

        self.Convolution.W = weight

        if bias is not None:
            assert isinstance(bias, np.ndarray), f'Expect <bias> as <np.ndarray>, is {type(bias)}.'
            
            _t_b = type(bias.take(0))
            assert _t_b == self.dtype, f'Expect <bias.dtype> as {self.dtype}, is {_t_b}.'

            self.bias = bias

            self.Convolution.b = bias


    def _weight_bias(self):
        return [self.weight, self.bias]


    def weight(self):
        return self._weight_bias()[0]


    def bias(self):
        return self._weight_bias()[1]


    def forward(self, X):
        torch_Tensor = False
        device = None

        if isinstance(X, torch.Tensor):
            _X = X.numpy(force = True) 

            torch_Tensor = True
            device = X.device.type
        elif isinstance(X, np.ndarray):
            _X = X
        else:
            raise TypeError(f'Expect <X> as [<torch.Tensor> | <np.ndarray>], is {type(X)}.')

        assert _X.ndim == 4, f'Expect <X.shape> as (N, C, H, W), is {X.shape}.'

        N, C, H, W = _X.shape

        assert C == self.weight.shape[1], f'Expect <X.shape[1]> as {self.Convolution.W.shape[1]}, is {C}.'

        if self.device == 'cpu':
            if type(_X.take(0)) != self.dtype:
                if self.nocast:
                    raise TypeError(f'Expect {self.dtype}, is {type(_X.take(0))}.')

                _X = astype(_X, self.dtype)

            _Y = self.Convolution.forward(_X)
        elif self.device == 'fpga':
            if type(_X.take(0)) != self.dtype:
                if not type(_X.take(0)) in POSIT_TO_UNSIGNED_MAPPING[self.dtype]:
                    if self.nocast: 
                        raise TypeError(f'Expect {self.dtype} or {POSIT_TO_UNSIGNED_MAPPING[self.dtype]}, is {type(_X.take(0))}.')

                    _X = tobin(astype(_X, self.dtype))
            else:
                _X = tobin(_X)

            Hout = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
            Wout = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

            _X = np.pad(_X, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), constant_values = type(_X.take(0))(0))

            A = np.empty([N, Hout * Wout, self.in_channels * self.kernel_size[0] * self.kernel_size[1] + 1], dtype = type(_X.take(0)))

            for n in range(N):
                for i in range(Hout):
                    for j in range(Wout):
                        A[n, i * Wout + j, :-1] = _X[n, :, i * self.stride[0]:i * self.stride[0] + self.kernel_size[0], j * self.stride[1]:j * self.stride[1] + self.kernel_size[1]].flatten()

            A = A.reshape(N * Hout * Wout, -1)
            A[:, -1] = tobin(astype(np.ones((1)), self.dtype)).take(0)

            _Y = self.default_remoteaccel.GEMM(A, tobin(np.vstack((self.weight.reshape(self.out_channels, -1).T, self.bias))))

            _Y = _Y.reshape(N, Hout, Wout, self.out_channels).transpose(0, 3, 1, 2)

            if not self.nocast:
                _Y = frombin(_Y, self.dtype)

        if torch_Tensor:
            if issubclass(type(_Y.take(0)), (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)):
                _Y = astype(_Y, np.double)

            Y = torch.from_numpy(_Y).to(device)
        else:
            Y = _Y

        return Y


    @classmethod
    def from_float(cls, float_model):
        assert type(float_model) == cls._FLOAT_MODEL, f'Expect <float_model> as {_FLOAT_MODEL}, is {type(float_model)}.'
        assert hasattr(float_model, 'qconfig') and isinstance(float_model.qconfig, QConfig), f'Expect <float_model.qconfig> as {QConfig}.'

        weight_post_process = float_model.qconfig.weight()

        assert isinstance(weight_post_process, ObserverBase), f'Expect <weight_post_process> as <ObserverBase>, is {type(weight_post_process)}.'

        assert issubclass(weight_post_process.dtype, (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)), f'Expect <weight_post_process.dtype> as [softposit.posit8 | softposit.posit16 | softposit.posit32 | softposit.posit_2], is {weight_post_process.dtype}.'

        weight_post_process(float_model.weight)

        max_workers = 1
        if hasattr(float_model, 'max_workers') and type(float_model.max_workers) == int:
            max_workers = float_model.max_workers

        qdevice = 'cpu'
        if hasattr(float_model, 'qdevice') and float_model.qdevice in ['cpu', 'fpga']:
            qdevice = float_model.qdevice

        fpga_host = '127.0.0.1'
        if hasattr(float_model, 'fpga_host') and type(float_model.fpga_host) == str:
            fpga_host = float_model.fpga_host

        fpga_port = 8080
        if hasattr(float_model, 'fpga_port') and type(float_model.fpga_port) == int:
            fpga_port = float_model.fpga_port

        fpga_conf = 'KV260 4 4 8 2 128 8'
        if hasattr(float_model, 'fpga_conf') and type(float_model.fpga_conf) == str:
            fpga_conf = float_model.fpga_conf

        nocast = False
        if hasattr(float_model, 'nocast') and type(float_model.nocast) == bool:
            nocast = float_model.nocast

        quantized_model = cls(
                float_model.in_channels,
                float_model.out_channels,
                float_model.kernel_size,
                float_model.stride,
                float_model.padding,
                float_model.dilation,
                float_model.groups,
                float_model.bias is not None,
                float_model.padding_mode,
                device = qdevice, dtype = weight_post_process.dtype,
                fpga_host = fpga_host, fpga_port = fpga_port, fpga_conf = fpga_conf,
                nocast = nocast,
                max_workers = max_workers
                )

        weight = astype(float_model.weight.float(), dtype = weight_post_process.dtype)
        bias = astype(float_model.bias.float(), dtype = weight_post_process.dtype) if float_model.bias is not None else astype(np.zeros(float_model.out_channels), dtype = weight_post_process.dtype)

        quantized_model.set_weight_bias(weight, bias)

        return quantized_model


    @classmethod
    def from_observed(cls, observed_model):
        return observed_model


class ReLU(nn.Module):
    _FLOAT_MODEL = nn.ReLU

    def __init__(
            self,
            inplace = False,
            dtype = None, device = None,
            fpga_host = '127.0.0.1', fpga_port = 8080, fpga_conf = 'KV260 4 4 8 2 128 8',
            nocast = False,
            max_workers = 1):
        super(ReLU, self).__init__()

        self.dtype = dtype
        self.device = device

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.fpga_host = fpga_host
        self.fpga_port = fpga_port
        self.fpga_conf = fpga_conf
        self.nocast = nocast
        self.max_workers = max_workers

        Part = str(fpga_conf.split(' ')[0])
        R = int(fpga_conf.split(' ')[1])
        C = int(fpga_conf.split(' ')[2])
        N = int(fpga_conf.split(' ')[3])
        Es = int(fpga_conf.split(' ')[4])
        QSize = int(fpga_conf.split(' ')[5])
        Depth = int(fpga_conf.split(' ')[6])

        if device == 'fpga':
            assert N == POSIT_TO_N_ES_MAPPING[dtype]['N']
            assert Es == POSIT_TO_N_ES_MAPPING[dtype]['Es']

        self.Part = Part
        self.R = R
        self.C = C
        self.N = N
        self.Es = Es
        self.QSize = QSize
        self.Depth = Depth

        self.Relu = Relu()


    def forward(self, X):
        torch_Tensor = False
        device = None

        if isinstance(X, torch.Tensor):
            _X = X.numpy(force = True)
            torch_Tensor = True
            device = X.device.type
        elif isinstance(X, np.ndarray):
            _X = X
        else:
            raise TypeError(f'Expect <X> as [<torch.Tensor> | <np.ndarray>], is {type(X)}.')

        if self.device == 'fpga':
            if type(_X.take(0)) != self.dtype:
                if not type(_X.take(0)) in POSIT_TO_UNSIGNED_MAPPING[self.dtype]:
                    if self.nocast:
                        raise TypeError(f'Expect {self.dtype} or {POSIT_TO_UNSIGNED_MAPPING[self.dtype]}, is {type(_X.take(0))}.')

                    _X = tobin(astype(_X, self.dtype))
            else:
                _X = tobin(_X)

            _Y = np.where(_X & 1 << (self.N - 1), type(_X.take(0))(0), _X)
        else:
            if type(X.take(0)) != self.dtype:
                if self.nocast:
                    raise TypeError(f'Expect {self.dtype}, is {type(X.take(0))}.')

                _X = astype(_X, self.dtype)

            _Y = self.Relu.forward(_X)

        if torch_Tensor:
            if issubclass(type(_Y.take(0)), (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)):
                _Y = astype(_Y, np.double)

            Y = torch.from_numpy(_Y).to(device)
        else:
            Y = _Y

        return Y


    @classmethod
    def from_float(cls, float_model):
        assert type(float_model) == cls._FLOAT_MODEL, f'Expect <float_model> as {_FLOAT_MODEL}, is {type(float_model)}.'
        assert hasattr(float_model, 'qconfig') and isinstance(float_model.qconfig, QConfig), f'Expect <float_model.qconfig> as {QConfig}.'

        weight_post_process = float_model.qconfig.weight()

        assert isinstance(weight_post_process, ObserverBase), f'Expect <weight_post_process> as <ObserverBase>, is {type(weight_post_process)}.'

        assert issubclass(weight_post_process.dtype, (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)), f'Expect <weight_post_process.dtype> as [softposit.posit8 | softposit.posit16 | softposit.posit32 | softposit.posit_2], is {weight_post_process.dtype}.'

        max_workers = 1
        if hasattr(float_model, 'max_workers') and type(float_model.max_workers) == int:
            max_workers = float_model.max_workers

        qdevice = 'cpu'
        if hasattr(float_model, 'qdevice') and float_model.qdevice in ['cpu', 'fpga']:
            qdevice = float_model.qdevice

        fpga_host = '127.0.0.1'
        if hasattr(float_model, 'fpga_host') and type(float_model.fpga_host) == str:
            fpga_host = float_model.fpga_host

        fpga_port = 8080
        if hasattr(float_model, 'fpga_port') and type(float_model.fpga_port) == int:
            fpga_port = float_model.fpga_port

        fpga_conf = 'KV260 4 4 8 2 128 8'
        if hasattr(float_model, 'fpga_conf') and type(float_model.fpga_conf) == str:
            fpga_conf = float_model.fpga_conf

        nocast = False
        if hasattr(float_model, 'nocast') and type(float_model.nocast) == bool:
            nocast = float_model.nocast

        return cls(
                float_model.inplace,
                dtype = weight_post_process.dtype, device = qdevice,
                fpga_host = fpga_host, fpga_port = fpga_port, fpga_conf = fpga_conf,
                nocast = nocast,
                max_workers = max_workers)


    @classmethod
    def from_observed(cls, observed_model):
        return observed_model


class MaxPool2d(nn.Module):
    _FLOAT_MODEL = nn.MaxPool2d

    def __init__(
            self,
            kernel_size, stride = None, padding = 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False,
            dtype = None, device = None,
            fpga_host = '127.0.0.1', fpga_port = 8080, fpga_conf = 'KV260 4 4 8 2 128 8',
            nocast = False,
            max_workers = 1):
        super(MaxPool2d, self).__init__()

        self.dtype = dtype
        self.device = device

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.fpga_host = fpga_host
        self.fpga_port = fpga_port
        self.fpga_conf = fpga_conf
        self.nocast = nocast
        self.max_workers = max_workers

        Part = str(fpga_conf.split(' ')[0])
        R = int(fpga_conf.split(' ')[1])
        C = int(fpga_conf.split(' ')[2])
        N = int(fpga_conf.split(' ')[3])
        Es = int(fpga_conf.split(' ')[4])
        QSize = int(fpga_conf.split(' ')[5])
        Depth = int(fpga_conf.split(' ')[6])

        if device == 'fpga':
            assert N == POSIT_TO_N_ES_MAPPING[dtype]['N']
            assert Es == POSIT_TO_N_ES_MAPPING[dtype]['Es']

        self.Part = Part
        self.R = R
        self.C = C
        self.N = N
        self.Es = Es
        self.QSize = QSize
        self.Depth = Depth

        self.Pooling = Pooling(pool_h = kernel_size, pool_w = kernel_size, stride = stride, pad = padding)


    def forward(self, X):
        torch_Tensor = False
        device = None

        if isinstance(X, torch.Tensor):
            _X = X.numpy(force = True)
            torch_Tensor = True
            device = X.device.type
        elif isinstance(X, np.ndarray):
            _X = X
        else:
            raise TypeError(f'Expect <X> as [<torch.Tensor> | <np.ndarray>], is {type(X)}.')

        _Y = self.Pooling.forward(_X)

        if torch_Tensor:
            if issubclass(type(_Y.take(0)), (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)):
                _Y = astype(_Y, np.double)

            Y = torch.from_numpy(_Y).to(device)
        else:
            Y = _Y

        return Y


    @classmethod
    def from_float(cls, float_model):
        assert type(float_model) == cls._FLOAT_MODEL, f'Expect <float_model> as {_FLOAT_MODEL}, is {type(float_model)}.'
        assert hasattr(float_model, 'qconfig') and isinstance(float_model.qconfig, QConfig), f'Expect <float_model.qconfig> as {QConfig}.'

        weight_post_process = float_model.qconfig.weight()

        assert isinstance(weight_post_process, ObserverBase), f'Expect <weight_post_process> as <ObserverBase>, is {type(weight_post_process)}.'

        assert issubclass(weight_post_process.dtype, (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)), f'Expect <weight_post_process.dtype> as [softposit.posit8 | softposit.posit16 | softposit.posit32 | softposit.posit_2], is {weight_post_process.dtype}.'

        max_workers = 1
        if hasattr(float_model, 'max_workers') and type(float_model.max_workers) == int:
            max_workers = float_model.max_workers

        qdevice = 'cpu'
        if hasattr(float_model, 'qdevice') and float_model.qdevice in ['cpu', 'fpga']:
            qdevice = float_model.qdevice

        fpga_host = '127.0.0.1'
        if hasattr(float_model, 'fpga_host') and type(float_model.fpga_host) == str:
            fpga_host = float_model.fpga_host

        fpga_port = 8080
        if hasattr(float_model, 'fpga_port') and type(float_model.fpga_port) == int:
            fpga_port = float_model.fpga_port

        fpga_conf = 'KV260 4 4 8 2 128 8'
        if hasattr(float_model, 'fpga_conf') and type(float_model.fpga_conf) == str:
            fpga_conf = float_model.fpga_conf

        nocast = False
        if hasattr(float_model, 'nocast') and type(float_model.nocast) == bool:
            nocast = float_model.nocast

        return cls(
                float_model.kernel_size, float_model.stride, float_model.padding, float_model.dilation, float_model.return_indices, float_model.ceil_mode,
                dtype = weight_post_process.dtype, device = qdevice,
                fpga_host = fpga_host, fpga_port = fpga_port, fpga_conf = fpga_conf,
                nocast = nocast,
                max_workers = max_workers)


    @classmethod
    def from_observed(cls, observed_model):
        return observed_model
