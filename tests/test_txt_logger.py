import os
import tempfile

import numpy as np
import torch

from data_vis import TxtLogger


def main():
    # base_dir = os.path.join(tempfile.gettempdir(), "data_vis_txt_demo")
    # logger = TxtLogger(base_dir=base_dir, enable=True)
    # print(f"Logging txt files into: {base_dir}")

    logger = TxtLogger()

    for i in range(5):
        logger.log_data("scalar", float(i))
        logger.log_data("vec", np.array([i, i + 1, i + 2], dtype=np.float32))
        logger.log_data("tensor2d", torch.tensor([[i, i + 1], [i + 2, i + 3]], dtype=torch.float32), batch_idx=0)

    print("Done. Generated files:")
    for name in ("scalar.txt", "vec.txt", "tensor2d.txt"):
        path = os.path.join(logger.base_dir, name)
        print(f"- {path} (exists={os.path.exists(path)})")


if __name__ == "__main__":
    main()
