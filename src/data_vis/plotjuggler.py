import socket
import json
import time
import torch
import numpy as np

class PlotJugglerUDP:
    def __init__(self, host="127.0.0.1", port=5005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.counter = 0

    def send_dict(self, data: dict, add_timestamp=True):
        payload = dict(data)
        if add_timestamp:
            payload["ts"] = time.time()
        msg = json.dumps(payload).encode("utf-8")
        self.sock.sendto(msg, self.addr)
        self.counter += 1
        if self.counter % 100 == 0:
            print(f"Sent {self.counter} messages to PlotJuggler.")


    def send_tensor(self, name: str, x: torch.Tensor, add_timestamp=True):
        # 切图，避免把训练图带出来
        x = x.detach()

        # 如果在 GPU，先搬到 CPU
        x = x.cpu()

        data = {}

        if x.ndim == 0:
            data[name] = float(x.item())
        elif x.ndim == 1:
            for i, v in enumerate(x):
                data[f"{name}/{i}"] = float(v.item())
        elif x.ndim == 2:
            # 常见情况：[batch, dim]
            # 为了看单个样本，默认发 batch 第 0 个
            row0 = x[0]
            for i, v in enumerate(row0):
                data[f"{name}/{i}"] = float(v.item())
        else:
            # 更高维先拍平
            flat = x.reshape(-1)
            for i, v in enumerate(flat):
                data[f"{name}/{i}"] = float(v.item())

        self.send_dict(data, add_timestamp=add_timestamp)

    def send_array(self, name, x, add_timestamp=True):
        x = np.asarray(x)
        data = {}

        if x.ndim == 0:
            data[name] = float(x.item())
        else:
            flat = x.reshape(-1)
            for i, v in enumerate(flat):
                data[f"{name}/{i}"] = float(v)

        self.send_dict(data, add_timestamp=add_timestamp)