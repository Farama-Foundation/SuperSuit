import multiprocessing as mp
import numpy as np


class SharedArray:
    def __init__(self, shape, dtype):
        self.shared_arr = mp.Array(np.ctypeslib.as_ctypes_type(dtype), int(np.prod(shape)), lock=False)
        self.dtype = dtype
        self.shape = shape
        self._set_np_arr()

    def _set_np_arr(self):
        self.np_arr = np.frombuffer(self.shared_arr, dtype=self.dtype).reshape(self.shape)

    def __getstate__(self):
        return (self.shared_arr, self.dtype, self.shape)

    def __setstate__(self, state):
        self.shared_arr = state
        self._set_np_arr()
