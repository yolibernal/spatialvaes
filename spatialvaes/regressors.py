from typing import Sequence

import torch
from torch import nn

from spatialvaes.encoders import ImageEncoder


class ImageRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: Sequence[int],
        fully_connected_dims: Sequence[int],
        output_dim: int,
        conv_class=nn.Conv2d,
    ) -> None:
        super().__init__()

        self.encoder = ImageEncoder(in_channels=in_channels, hidden_dims=hidden_dims)

        modules = []
        modules.append(nn.Flatten())
        for i in range(len(fully_connected_dims)):
            if i == 0:
                module_in_channels = hidden_dims[-1]
                module_out_channels = fully_connected_dims[i]
            else:
                module_in_channels = fully_connected_dims[i - 1]
                module_out_channels = fully_connected_dims[i]
            modules.append(nn.Linear(module_in_channels, module_out_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(fully_connected_dims[-1], output_dim))
        self.regressor = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.regressor(x)
        return x
