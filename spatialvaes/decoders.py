from typing import Sequence

import torch
from torch import nn

from spatialvaes.layers import CoordConv2d


class ImageDecoder(nn.Module):
    def __init__(self, out_channels, hidden_dims: Sequence[int]) -> None:
        super().__init__()

        modules = []
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                module_in_channels = hidden_dims[-1]
                module_out_channels = out_channels
            else:
                module_in_channels = hidden_dims[i]
                module_out_channels = hidden_dims[i + 1]
            modules.append(
                nn.ConvTranspose2d(
                    module_in_channels,
                    module_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            modules.append(nn.BatchNorm2d(module_out_channels))
            if i == len(hidden_dims) - 1:
                modules.append(nn.Sigmoid())
            else:
                modules.append(nn.ReLU())
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x


class ImageUpscaleDecoder(nn.Module):
    def __init__(
        self, out_channels, hidden_dims: Sequence[int], conv_class=nn.Conv2d
    ) -> None:
        super().__init__()

        modules = []
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                module_in_channels = hidden_dims[-1]
                module_out_channels = out_channels
            else:
                module_in_channels = hidden_dims[i]
                module_out_channels = hidden_dims[i + 1]
            modules.append(nn.Upsample(scale_factor=2))
            modules.append(
                conv_class(
                    module_in_channels,
                    module_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if i == len(hidden_dims) - 1:
                modules.append(nn.Sigmoid())
            else:
                modules.append(nn.ReLU())
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x


# class ImageSingleCoordConvDecoder(nn.Module):
#     def __init__(self, out_channels, hidden_dims: Sequence[int], conv_class=nn.Conv2d) -> None:
#         super().__init__()

#         modules = []
#         modules.append(nn.Upsample(scale_factor=2 ** len(hidden_dims)))
#         for i in range(len(hidden_dims)):
#             if i == 0:
#                 conv_class = CoordConv2d
#             else:
#                 conv_class = nn.Conv2d

#             if i == len(hidden_dims) - 1:
#                 module_in_channels = hidden_dims[-1]
#                 module_out_channels = out_channels
#             else:
#                 module_in_channels = hidden_dims[i]
#                 module_out_channels = hidden_dims[i + 1]
#             modules.append(
#                 conv_class(
#                     module_in_channels, module_out_channels, kernel_size=3, stride=1, padding=1
#                 )
#             )
#             if i == len(hidden_dims) - 1:
#                 modules.append(nn.Sigmoid())
#             else:
#                 modules.append(nn.ReLU())
#         self.decoder = nn.Sequential(*modules)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.decoder(x)
#         return x


class ImageSingleCoordConvDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_dims: Sequence[int],
        hidden_dims_after_first_coordconv: Sequence[int],
        conv_class_after_first_coordconv=nn.Conv2d,
    ) -> None:
        super().__init__()

        modules = []
        for i in range(len(hidden_dims) - 1):
            if i == len(hidden_dims) - 2:
                # conv_class = CoordConv2d
                conv_class = nn.Conv2d
            else:
                conv_class = nn.Conv2d

            module_in_channels = hidden_dims[i]
            module_out_channels = hidden_dims[i + 1]

            modules.append(nn.Upsample(scale_factor=2))
            modules.append(
                conv_class(
                    module_in_channels,
                    module_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            modules.append(nn.ReLU())
        modules.append(nn.Upsample(scale_factor=2))
        for i in range(len(hidden_dims_after_first_coordconv)):
            if i == len(hidden_dims_after_first_coordconv) - 1:
                module_in_channels = hidden_dims_after_first_coordconv[0]
                module_out_channels = out_channels
            else:
                module_in_channels = hidden_dims_after_first_coordconv[-(i + 1)]
                module_out_channels = hidden_dims_after_first_coordconv[-(i + 2)]
            modules.append(
                conv_class_after_first_coordconv(
                    module_in_channels,
                    module_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if i == len(hidden_dims_after_first_coordconv) - 1:
                modules.append(nn.Sigmoid())
            else:
                modules.append(nn.ReLU())

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x


class SpatialBroadcastDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_dims: Sequence[int],
        in_resolution=128,
        conv_class=nn.Conv2d,
        kernel_size=3,
        num_coordinate_channel_pairs=1,
    ) -> None:
        super().__init__()

        self.in_resolution = in_resolution
        x = torch.linspace(-1, 1, in_resolution)
        y = torch.linspace(-1, 1, in_resolution)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        # Add as constant, with extra dims for N and C
        self.register_buffer("x_grid", (x_grid.view((1, 1) + x_grid.shape)).clone())
        self.register_buffer("y_grid", (y_grid.view((1, 1) + y_grid.shape)).clone())

        self.num_coordinate_channel_pairs = num_coordinate_channel_pairs

        # modules = [
        #     nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
        # ]

        modules = []
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                module_in_channels = hidden_dims[-1]
                module_out_channels = out_channels
            else:
                module_in_channels = hidden_dims[i]
                module_out_channels = hidden_dims[i + 1]
            modules.append(
                conv_class(
                    module_in_channels,
                    module_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                )
            )
            if i == len(hidden_dims) - 1:
                pass
                # modules.append(nn.Sigmoid())
            else:
                modules.append(nn.ReLU())
        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        # z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.in_resolution, self.in_resolution)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        # x = torch.cat(
        #     (
        #         self.x_grid.expand(batch_size, -1, -1, -1),
        #         self.y_grid.expand(batch_size, -1, -1, -1),
        #         z,
        #     ),
        #     dim=1,
        # )
        coordinate_channels = torch.cat(
            (
                self.x_grid.expand(batch_size, -1, -1, -1),
                self.y_grid.expand(batch_size, -1, -1, -1),
            ),
            dim=1,
        ).repeat((1, self.num_coordinate_channel_pairs, 1, 1))
        x = torch.cat(
            (
                coordinate_channels,
                z,
            ),
            dim=1,
        )

        x = self.decoder(x)

        return x
