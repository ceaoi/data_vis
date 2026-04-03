from .plotjuggler import PlotJugglerUDP
from .txt_logger import TxtLogger

plotjuggler = PlotJugglerUDP(host="127.0.0.1", port=5005)
txt_logger = TxtLogger()

__all__ = ["plotjuggler", "txt_logger"]