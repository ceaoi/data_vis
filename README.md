
# 安装
```bash
git clone https://github.com/ceaoi/data_vis.git
cd ./data_vis
pip install -e .
```

# 使用示例：


## data to plotjuggler
```python
from data_vis import PlotJugglerUDP

plotjuggler = PlotJugglerUDP(host="127.0.0.1", port=5005)

plotjuggler.send_dict({"step": i, "loss": 1.0 / (i + 1)})
plotjuggler.send_tensor("feat", torch.tensor([i, i * 2.0, i * 3.0]))
plotjuggler.send_array("arr", np.array([i, i + 0.5, i + 1.0], dtype=np.float32))
```

## data to txt
```python
from data_vis import TxtLogger

txt_logger = TxtLogger()

logger.log_data("scalar", float(i))
logger.log_data("vec", np.array([i, i + 1, i + 2], dtype=np.float32))
logger.log_data("tensor2d", torch.tensor([[i, i + 1], [i + 2, i + 3]], dtype=torch.float32), batch_idx=0)
```