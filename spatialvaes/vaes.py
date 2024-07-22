from typing import Any, Dict, Sequence

import torch
from torch import nn

from spatialvaes.decoders import ImageDecoder, SpatialBroadcastDecoder
from spatialvaes.encoders import ImageEncoder


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()

        # modules = nn.Sequential(
        #     nn.Linear(in_features=1024, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=20),
        # )

        modules = []
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                module_in_channels = in_dim
                module_out_channels = hidden_dims[i]
            elif i == len(hidden_dims):
                module_in_channels = hidden_dims[i - 1]
                module_out_channels = out_dim
            else:
                module_in_channels = hidden_dims[i - 1]
                module_out_channels = hidden_dims[i]
            modules.append(nn.Linear(module_in_channels, module_out_channels))
            if i != len(hidden_dims):
                modules.append(nn.ReLU())
            # modules.append(nn.BatchNorm1d(module_out_channels))
            # modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        # return torch.chunk(x, 2, dim=1)

        return x


class ChunkedMLP(MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        return torch.chunk(x, 2, dim=1)


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_mu: nn.Module,
        encoder_logvar: nn.Module,
        # encoder_mu_logvar: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.encoder_mu = encoder_mu
        self.encoder_logvar = encoder_logvar
        # self.encoder_mu_logvar = encoder_mu_logvar

        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        # mu, logvar = self.encoder_mu_logvar(features)
        mu = self.encoder_mu(features)
        logvar = self.encoder_logvar(features)
        z = self.reparameterize(mu, logvar)
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_hat = self.decoder(z)
        kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        return x_hat, kl_divergence

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class ImageVAE(VAE):
    def __init__(
        self,
        in_resolution: int,
        in_channels: int,
        hidden_dims_encoder: Sequence[int],
        hidden_dims_decoder: Sequence[int],
        fc_dims: Sequence[int],
        latent_dim: int = 128,
    ) -> None:
        encoder = ImageEncoder(in_channels, hidden_dims_encoder)

        in_dim = hidden_dims_encoder[-1] * (in_resolution // 2 ** len(hidden_dims_encoder)) ** 2
        encoder_mu = MLP(in_dim=in_dim, out_dim=latent_dim, hidden_dims=fc_dims)
        encoder_logvar = MLP(in_dim=in_dim, out_dim=latent_dim, hidden_dims=fc_dims)
        decoder = ImageDecoder(in_channels, [latent_dim] + hidden_dims_decoder)
        super().__init__(encoder, encoder_mu, encoder_logvar, decoder)

    @classmethod
    def from_config(cls, conf: Dict[str, Any]) -> "ImageVAE":
        return cls(
            in_resolution=conf["in_resolution"],
            in_channels=conf["in_channels"],
            hidden_dims_encoder=conf["hidden_dims_encoder"],
            hidden_dims_decoder=conf["hidden_dims_decoder"],
            fc_dims=conf["fc_dims"],
            latent_dim=conf["latent_dim"],
        )


class SpatialBroadcastVAE(VAE):
    def __init__(
        self,
        in_resolution: int,
        in_channels: int,
        hidden_dims_encoder: Sequence[int],
        hidden_dims_decoder: Sequence[int],
        fc_dims: Sequence[int],
        kernel_size_encoder: int = 3,
        kernel_size_decoder: int = 3,
        latent_dim: int = 128,
        num_coordinate_channel_pairs: int = 1,
    ) -> None:
        encoder = ImageEncoder(in_channels, hidden_dims_encoder, kernel_size=kernel_size_encoder)

        in_dim = hidden_dims_encoder[-1] * (in_resolution // 2 ** len(hidden_dims_encoder)) ** 2
        # print(f"in_dim: {in_dim}")
        # encoder_mu_logvar = ChunkedMLP(in_dim=in_dim, out_dim=latent_dim * 2, hidden_dims=fc_dims)
        encoder_mu = MLP(in_dim=in_dim, out_dim=latent_dim, hidden_dims=fc_dims)
        encoder_logvar = MLP(in_dim=in_dim, out_dim=latent_dim, hidden_dims=fc_dims)
        decoder = SpatialBroadcastDecoder(
            in_channels,
            [latent_dim + 2 * num_coordinate_channel_pairs] + hidden_dims_decoder,
            in_resolution=in_resolution,
            kernel_size=kernel_size_decoder,
            num_coordinate_channel_pairs=num_coordinate_channel_pairs,
        )
        super().__init__(encoder, encoder_mu, encoder_logvar, decoder)
        # super().__init__(encoder, encoder_mu_logvar, decoder)

    @classmethod
    def from_config(cls, conf: Dict[str, Any]) -> "SpatialBroadcastVAE":
        return cls(
            in_resolution=conf["in_resolution"],
            in_channels=conf["in_channels"],
            hidden_dims_encoder=conf["hidden_dims_encoder"],
            hidden_dims_decoder=conf["hidden_dims_decoder"],
            fc_dims=conf["fc_dims"],
            latent_dim=conf["latent_dim"],
            kernel_size_encoder=conf["kernel_size_encoder"],
            kernel_size_decoder=conf["kernel_size_decoder"],
            num_coordinate_channel_pairs=conf["num_coordinate_channel_pairs"],
        )
