from typing import Sequence

import torch
from torch import nn

from spatialvaes.decoders import (
    ImageDecoder,
    ImageSingleCoordConvDecoder,
    ImageUpscaleDecoder,
)
from spatialvaes.encoders import ImageEncoder, ImageSingleCoordConvEncoder
from spatialvaes.layers import CoordConv2d


class Bottleneck(nn.Module):
    """Bottleneck module with fully connected layers and ReLU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()

        modules = []
        modules.append(nn.Flatten())
        for i in range(len(hidden_dims)):
            if i == 0:
                module_in_channels = in_dim
                module_out_channels = hidden_dims[i]
            elif i == len(hidden_dims) - 1:
                module_in_channels = hidden_dims[i - 1]
                module_out_channels = out_dim
            else:
                module_in_channels = hidden_dims[i - 1]
                module_out_channels = hidden_dims[i]
            modules.append(nn.Linear(module_in_channels, module_out_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Unflatten(dim=1, unflattened_size=(out_dim, 1, 1)))
        self.bottleneck = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(x)
        return x


class AE(nn.Module):
    """Autoencoder with encoder, bottleneck, and decoder."""

    def __init__(self, encoder: nn.Module, bottleneck: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class ImageAE(AE):
    """Autoencoder using image encoder and decoder"""

    def __init__(
        self,
        in_resolution: int,
        in_channels: int,
        hidden_dims: Sequence[int],
        bottleneck_dims: Sequence[int],
    ) -> None:
        encoder = ImageEncoder(in_channels, hidden_dims)
        bottleneck = Bottleneck(
            in_dim=hidden_dims[-1], out_dim=hidden_dims[-1], hidden_dims=bottleneck_dims
        )
        decoder = ImageDecoder(in_channels, hidden_dims[::-1])
        super().__init__(encoder, bottleneck, decoder)


class ImageUpscaleAE(ImageAE):
    """Image autoencoder using upscale deocder"""

    def __init__(
        self,
        in_resolution: int,
        in_channels: int,
        hidden_dims: Sequence[int],
        bottleneck_dims: Sequence[int],
    ) -> None:
        super().__init__(in_resolution, in_channels, hidden_dims, bottleneck_dims)

        self.decoder = ImageUpscaleDecoder(in_channels, hidden_dims[::-1])


class ImageCoordConvAE(ImageAE):
    """Image autoencoder using CoordConv decoder"""

    def __init__(
        self,
        in_resolution: int,
        in_channels: int,
        hidden_dims: Sequence[int],
        bottleneck_dims: Sequence[int],
    ) -> None:
        super().__init__(in_resolution, in_channels, hidden_dims, bottleneck_dims)

        self.decoder = ImageUpscaleDecoder(in_channels, hidden_dims[::-1], conv_class=CoordConv2d)


class ImageSingleCoordConvAE(ImageAE):
    """Image autoencoder with decoder using single CoordConv layer"""

    def __init__(
        self,
        in_resolution: int,
        in_channels: int,
        hidden_dims: Sequence[int],
        bottleneck_dims: Sequence[int],
        hidden_dims_after_first_coordconv: Sequence[int],
    ) -> None:
        super().__init__(in_resolution, in_channels, hidden_dims, bottleneck_dims)

        self.encoder = ImageSingleCoordConvEncoder(in_channels, hidden_dims)
        self.decoder = ImageSingleCoordConvDecoder(
            in_channels,
            hidden_dims[::-1],
            hidden_dims_after_first_coordconv=hidden_dims_after_first_coordconv,
        )
