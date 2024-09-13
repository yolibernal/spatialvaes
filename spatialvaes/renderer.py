from typing import Sequence

import torch
from torch import nn

from spatialvaes.decoders import (
    ImageDecoder,
    ImageSingleCoordConvDecoder,
    ImageUpscaleDecoder,
)
from spatialvaes.layers import CoordConv2d


class ImageRenderer(nn.Module):
    """Render image from low-dimensional (cartesian) representation of object position."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()

        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0] // 4),
            nn.Linear(hidden_dims[0] // 4, hidden_dims[0] // 2),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0]),
            nn.Unflatten(dim=1, unflattened_size=(hidden_dims[0], 1, 1)),
        )
        self.decoder = ImageDecoder(out_channels=3, hidden_dims=hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.decoder(x)
        return x


class ImageUpscaleRenderer(ImageRenderer):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__(input_dim, hidden_dims)

        self.decoder = ImageUpscaleDecoder(
            out_channels=3, hidden_dims=hidden_dims, conv_class=nn.Conv2d
        )


class ImageCoordConvRenderer(ImageRenderer):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__(input_dim, hidden_dims)

        self.decoder = ImageUpscaleDecoder(
            out_channels=3, hidden_dims=hidden_dims, conv_class=CoordConv2d
        )


class ImageSingleCoordConvRenderer(ImageRenderer):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        hidden_dims_after_first_coordconv: Sequence[int],
    ) -> None:
        super().__init__(input_dim, hidden_dims)

        self.decoder = ImageSingleCoordConvDecoder(
            out_channels=3,
            hidden_dims=hidden_dims,
            hidden_dims_after_first_coordconv=hidden_dims_after_first_coordconv,
        )
