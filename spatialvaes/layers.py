import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from torch import nn


def add_coords(x):
    """
    Adds coordinate encodings to a tensor.

    Parameters:
    -----------
    x: torch.Tensor of shape (b, c, h, w)
        Input tensor

    Returns:
    --------
    augmented_x: torch.Tensor of shape (b, c+2, h, w)
        Input tensor augmented with two new channels with positional encodings
    """

    b, c, h, w = x.shape
    coords_h = torch.linspace(-1, 1, h, device=x.device)[:, None].expand(b, 1, h, w)
    coords_w = torch.linspace(-1, 1, w, device=x.device).expand(b, 1, h, w)
    return torch.cat([x, coords_h, coords_w], 1)


class CoordConv2d(nn.Module):
    """
    Conv2d that adds coordinate encodings to the input (Liu et al., 2018)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(add_coords(x))
