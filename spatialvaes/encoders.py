from typing import Sequence

import torch
from torch import nn

from spatialvaes.layers import CoordConv2d


class ImageEncoder(nn.Module):
    """Encode image using convolutional layers."""

    def __init__(
        self, in_channels: int, hidden_dims: Sequence[int], conv_class=nn.Conv2d, kernel_size=3
    ) -> None:
        super().__init__()

        modules = []
        for i in range(len(hidden_dims)):
            if i == 0:
                module_in_channels = in_channels
                module_out_channels = hidden_dims[i]
            else:
                module_in_channels = hidden_dims[i - 1]
                module_out_channels = hidden_dims[i]
            modules.append(
                conv_class(
                    module_in_channels,
                    module_out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x


class ImageSingleCoordConvEncoder(nn.Module):
    """Encode image using convolutional layers with single CoordConv layer."""

    def __init__(self, in_channels: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()

        modules = []
        for i in range(len(hidden_dims)):
            if i == 0:
                module_in_channels = in_channels
                module_out_channels = hidden_dims[i]
                conv_class = CoordConv2d
            else:
                module_in_channels = hidden_dims[i - 1]
                module_out_channels = hidden_dims[i]
                conv_class = nn.Conv2d

            modules.append(
                conv_class(
                    module_in_channels, module_out_channels, kernel_size=3, stride=2, padding=1
                )
            )
            modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x
