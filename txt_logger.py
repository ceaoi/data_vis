import os
import numpy as np
import torch


class TxtLogger:
    def __init__(self, base_dir=None, enable=True):
        if base_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(cur_dir, "data_txt")

        self.base_dir = base_dir
        self.enable = enable
        os.makedirs(self.base_dir, exist_ok=True)

        # 记录哪些文件本轮程序已经初始化过
        self._initialized_files = set()
        self.counter = 0

    def _get_path(self, name: str):
        filename = f"{name}.txt"
        return os.path.join(self.base_dir, filename)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        elif isinstance(x, np.ndarray):
            x = x.astype(np.float32, copy=False)
        elif isinstance(x, (list, tuple)):
            x = np.asarray(x, dtype=np.float32)
        elif isinstance(x, (int, float, bool)):
            x = np.asarray([x], dtype=np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(x)}")
        return x

    def log_tensor(self, name: str, x, batch_idx=0, flatten=True, fmt="%.6f"):
        if not self.enable:
            return

        arr = self._to_numpy(x)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim >= 2:
            if batch_idx >= arr.shape[0]:
                raise IndexError(f"batch_idx={batch_idx} out of range for shape {arr.shape}")
            arr = arr[batch_idx]

        if flatten:
            arr = arr.reshape(-1)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        path = self._get_path(name)

        # 第一次写这个文件时，先清空
        if path not in self._initialized_files:
            with open(path, "w", encoding="utf-8"):
                pass
            self._initialized_files.add(path)

        # 后续一直追加新行
        with open(path, "ab") as f:
            np.savetxt(f, arr, fmt=fmt)

        self.counter += 1
        if self.counter % 100 == 0:
            print(f"Logged tensor '{name}' with shape {arr.shape} to {path}")