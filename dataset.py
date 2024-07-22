from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CausalImageDataset(Dataset):
    def __init__(
        self,
        npz_data,
        noise=None,
        size=None,
        mean=None,
        std=None,
        min=None,
        max=None,
        normalization="none",
        merge_before_after=True,
        flatten=False,
    ):
        self._x = npz_data["imgs"]
        self._z0 = npz_data["original_latents"][:, 0, :]
        self._z1 = npz_data["original_latents"][:, 1, :]
        self._intervention_labels = npz_data["intervention_labels"][:, np.newaxis]
        self._intervention_masks = npz_data["intervention_masks"]

        if size is not None:
            self._x = np.array(
                [
                    [np.array(Image.fromarray(x).resize(size, Image.NEAREST)) for x in self._x[i]]
                    for i in range(self._x.shape[0])
                ]
            )

        self._e0 = npz_data["epsilon"][:, 0, :]
        self._e1 = npz_data["epsilon"][:, 1, :]

        self.noise = noise

        self.size = size

        self.mean = mean
        self.std = std

        self.min = min
        self.max = max

        self.normalization = normalization
        self.flatten = flatten
        self.merge_before_after = merge_before_after

    def __getitem__(self, index):
        if self.merge_before_after:
            _index = index // 2
            _after = index % 2 == 1
            x = self._get_x(_index, _after)
            z = self._get_z(_index, _after)
            e = self._get_e(_index, _after)
            intervention_label = torch.LongTensor(self._intervention_labels[_index])  # (1,)
            intervention_mask = torch.BoolTensor(self._intervention_masks[_index])
            return x, z, e, intervention_label, intervention_mask

        x0 = self._get_x(index, False)
        x1 = self._get_x(index, True)
        z0 = self._get_z(index, False)
        z1 = self._get_z(index, True)
        intervention_label = torch.LongTensor(self._intervention_labels[index])  # (1,)
        intervention_mask = torch.BoolTensor(self._intervention_masks[index])

        if self.noise is not None and self.noise > 0.0:
            # noinspection PyTypeChecker
            x0 += self.noise * torch.randn_like(x0)
            x1 += self.noise * torch.randn_like(x1)

        e0 = self._get_e(index, False)
        e1 = self._get_e(index, True)
        return x0, x1, z0, z1, intervention_label, intervention_mask, e0, e1

    def __len__(self):
        if self.merge_before_after:
            return self._z0.shape[0] * 2
        return self._z0.shape[0]

    def _get_x(self, index, after=False):
        array = self._x[index, 1 if after else 0]
        if self.size is not None:
            array = np.array(Image.fromarray(array).resize(self.size))

        tensor = torch.FloatTensor(np.transpose(array, (2, 0, 1)))
        if self.normalization == "none":
            tensor = tensor / 255.0

        if self.normalization == "standard":
            if self.mean is None or self.std is None:
                raise ValueError("Mean and std must be provided for standard normalization")
            tensor = (tensor - torch.FloatTensor(self.mean).view(3, 1, 1)) / torch.FloatTensor(
                self.std
            ).view(3, 1, 1)

        if self.normalization == "minmax":
            if self.min is None or self.max is None:
                raise ValueError("Min and max must be provided for minmax normalization")
            tensor = (tensor - torch.FloatTensor(self.min).view(3, 1, 1)) / (
                torch.FloatTensor(self.max).view(3, 1, 1)
                - torch.FloatTensor(self.min).view(3, 1, 1)
            )

        if self.flatten:
            tensor = tensor.reshape(-1)

        return tensor

    def _get_z(self, index, after=False):
        array = self._z1[index] if after else self._z0[index]
        tensor = torch.FloatTensor(array)
        return tensor

    def _get_e(self, index, after=False):
        array = self._e1[index] if after else self._e0[index]
        tensor = torch.FloatTensor(array)
        return tensor


def load_data_parts(data_dir: str, tag: str):
    filenames = [Path(data_dir) / f"{tag}.npz"]

    data_parts = []
    for filename in filenames:
        assert filename.exists(), f"Dataset not found at {filename}. Consult README.md."
        data_parts.append(dict(np.load(filename)))

    data = {k: np.concatenate([data[k] for data in data_parts]) for k in data_parts[0]}
    return data
