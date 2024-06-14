import http.client

import numpy as np

class RemoteAccel:
    def __init__(self, host = '127.0.0.1', port = 8080, timeout = -1, Part = 'KV260', R = 4, C = 4, N = 8, Es = 2, QSize = 128, Depth = 8, num_threads = 1):
        super().__init__()

        if timeout <= 0:
            self.HTTPConnection = http.client.HTTPConnection(host, port)
        else:
            self.HTTPConnection = http.client.HTTPConnection(host, port, timeout)

        self.Part = None
        self.R = None
        self.C = None
        self.N = None
        self.Es = None
        self.QSize = None
        self.Depth = None

        self.num_threads = None

        self.load(Part, R, C, N, Es, QSize, Depth, num_threads)

    def load(self, Part, R, C, N, Es, QSize, Depth, num_threads, force = False):
        self.HTTPConnection.request('GET', f'/load?Part={Part}&R={R}&C={C}&N={N}&Es={Es}&QSize={QSize}&Depth={Depth}&num_threads={num_threads}&force={1 if force else 0}')

        HTTPResponse = self.HTTPConnection.getresponse()

        if HTTPResponse.status != 200:
            raise ConnectionError(HTTPResponse.reason)

        self.Part = Part
        self.R = R
        self.C = C
        self.N = N
        self.Es = Es
        self.QSize = QSize
        self.Depth = Depth

        self.num_threads = num_threads

    def GEMM(self, A, B):
        body = A.tobytes() + B.tobytes()

        headers = {
                'Content-Type': 'application/octet-stream',
                'Content-Length': len(body),
                'Ar': A.shape[0],
                'Ac': A.shape[1],
                'Br': B.shape[0],
                'Bc': B.shape[1]
                }

        self.HTTPConnection.request('POST', '/gemm', body, headers)

        HTTPResponse = self.HTTPConnection.getresponse()

        if HTTPResponse.status != 200:
            raise ConnectionError(HTTPResponse.reason)

        Yr, Yc = HTTPResponse.getheader('Yr'), HTTPResponse.getheader('Yc')
        Yr, Yc = int(Yr), int(Yc)

        N = 2 ** int(np.ceil(np.log2(self.N)))

        buffer = HTTPResponse.read()

        Y = np.frombuffer(buffer, np.__getattribute__(f'uint{N}')).reshape(Yr, Yc)

        return Y
