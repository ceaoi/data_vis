import time

import numpy as np
import torch

from data_vis import PlotJugglerUDP


def main():
    sender = PlotJugglerUDP(host="127.0.0.1", port=5005)

    print("Start sending UDP data to PlotJuggler...")
    print("Target: 127.0.0.1:5005")

    for i in range(20):
        sender.send_dict({"step": i, "loss": 1.0 / (i + 1)})
        sender.send_tensor("feat", torch.tensor([i, i * 2.0, i * 3.0]))
        sender.send_array("arr", np.array([i, i + 0.5, i + 1.0], dtype=np.float32))
        time.sleep(0.1)

    print(f"Done. Sent {sender.counter} messages.")


if __name__ == "__main__":
    main()
