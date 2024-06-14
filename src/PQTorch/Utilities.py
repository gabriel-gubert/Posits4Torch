import numpy as np

import softposit as sp

import torch

POSIT_TO_N_ES_MAPPING = {sp.posit8: {'N': 8, 'Es': 0}, sp.posit16: {'N': 16, 'Es': 1}, sp.posit32: {'N': 32, 'Es': 2}, sp.posit_2: {'N': 0, 'Es': 2}}

for i in range(1, 32):
    POSIT_TO_N_ES_MAPPING[sp.__getattribute__(f'posit{i}_2')] = {'N': i, 'Es': 2}

POSIT_TO_UNSIGNED_MAPPING = {sp.posit8: [np.uint8, np.uint16, np.uint32, np.uint64], sp.posit16: [np.uint16, np.uint32, np.uint64], sp.posit32: [np.uint32, np.uint64], sp.posit_2: [np.uint32, np.uint64]}

for i in range(1, 32):
    if 0 < i and i <= 8:
        POSIT_TO_UNSIGNED_MAPPING[sp.__getattribute__(f'posit{i}_2')] = [np.uint8, np.uint16, np.uint32, np.uint64]
    elif 8 < i and i <= 16:
        POSIT_TO_UNSIGNED_MAPPING[sp.__getattribute__(f'posit{i}_2')] = [np.uint16, np.uint32, np.uint64]
    elif 16 < i and i <= 32:
        POSIT_TO_UNSIGNED_MAPPING[sp.__getattribute__(f'posit{i}_2')] = [np.uint32, np.uint64]
    else:
        pass

POSIT_2_TO_RIGHT_SHIFT_MAPPING = {}

for i in range(1, 32):
    POSIT_2_TO_RIGHT_SHIFT_MAPPING[sp.__getattribute__(f'posit{i}_2')] = 32 - i

UNSIGNED_TO_POSIT_MAPPING = {np.uint8: [sp.posit8], np.uint16: [sp.posit8, sp.posit16], np.uint32: [sp.posit8, sp.posit16, sp.posit32, sp.posit_2], np.uint64: [sp.posit8, sp.posit16, sp.posit32, sp.posit_2]}

UNSIGNED_TO_POSIT_MAPPING[np.uint8].extend([sp.__getattribute__(f'posit{i}_2') for i in range(1, 8 + 1)])

UNSIGNED_TO_POSIT_MAPPING[np.uint16].extend([sp.__getattribute__(f'posit{i}_2') for i in range(1, 16 + 1)])

UNSIGNED_TO_POSIT_MAPPING[np.uint32].extend([sp.__getattribute__(f'posit{i}_2') for i in range(1, 32)])

UNSIGNED_TO_POSIT_MAPPING[np.uint64].extend([sp.__getattribute__(f'posit{i}_2') for i in range(1, 32)])

# UNSIGNED_TO_POSIT_MAPPING[int].extend([sp.__getattribute__(f'posit{i}_2') for i in range(1, 32)])

def gettype(N, Es):
    assert isinstance(N, int) and isinstance(Es, int), f'Expect (<N>, <Es>) as ({int}, {int}), is ({type(N)}, {type(Es)}).'

    try:
        if N == 8 and Es == 0:
            t = sp.posit8
        elif N == 16 and Es == 1:
            t = sp.posit16
        elif N == 32 and Es == 2:
            t = sp.posit32
        else:
            t = sp.__getattribute__(f'posit{N}_{Es}')

        return t
    except:
        raise RuntimeError("Expect (<N>, <Es>) as (N = 8, Es = 0), (N = 16, Es = 1) or (N = [1, 32], Es = 2), is (N = {N}, Es = {Es}).")

def astype(X, dtype, inplace = False):
    _X = X

    if isinstance(_X, torch.Tensor):
        _X = _X.numpy(force = True)

    assert isinstance(_X, np.ndarray), f'Expect <X> as {np.ndarray}, is {type(X)}.'

    if issubclass(dtype, (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)):
        _X = _X.astype(np.double)

    __X = np.empty_like(_X, dtype = dtype)

    for i in range(__X.size):
        __X.flat[i] = dtype(_X.flat[i])

    if inplace:
        X = __X

        return X

    return __X

def tobin(X, u_dtype = None, inplace = False):
    assert isinstance(X, np.ndarray), f'Expect <X> as {np.ndarray}, is {type(X)}.'

    p_dtype = type(X.take(0))

    assert issubclass(p_dtype, (sp.posit8, sp.posit16, sp.posit32, sp.posit_2)), f'Expect <p_dtype> as [sp.posit8 | sp.posit16 | sp.posit32 | sp.posit_2], is {p_dtype}.'

    if u_dtype is None:
        u_dtype = POSIT_TO_UNSIGNED_MAPPING[p_dtype][0]

    assert u_dtype in POSIT_TO_UNSIGNED_MAPPING[p_dtype], f'Expect <u_dtype> as {POSIT_TO_UNSIGNED_MAPPING[p_dtype]}, is {u_dtype}.'

    _X = np.empty_like(X, dtype = u_dtype)

    if issubclass(p_dtype, sp.posit_2):
        for i in range(_X.size):
            _X.flat[i] = u_dtype(X.flat[i].v.v >> POSIT_2_TO_RIGHT_SHIFT_MAPPING[p_dtype])
    else:
        for i in range(_X.size):
            _X.flat[i] = u_dtype(X.flat[i].v.v)

    if inplace:
        X = _X

        return X

    return _X

def frombin(X, p_dtype, inplace = False):
    assert isinstance(X, np.ndarray), f'Expect <X> as {np.ndarray}, is {type(X)}.'

    u_dtype = type(X.take(0))

    assert issubclass(u_dtype, (np.uint8, np.uint16, np.uint32, np.uint64)), f'Expect <u_dtype> as [np.uint8, np.uint16, np.uint32, np.uint64], is {u_dtype}.'

    assert p_dtype in UNSIGNED_TO_POSIT_MAPPING[u_dtype], f'Expect <p_dtype> as {UNSIGNED_TO_POSIT_MAPPING[u_dtype]}, is {p_dtype}.'

    _X = np.empty_like(X, dtype = p_dtype)

    for i in range(_X.size):
        _X.flat[i] = p_dtype(bits = X.flat[i].item())

    if inplace:
        X = _X

        return X

    return _X
