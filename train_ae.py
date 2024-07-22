from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import CausalImageDataset, load_data_parts
from spatialvaes.aes import (
    ImageAE,
    ImageCoordConvAE,
    ImageSingleCoordConvAE,
    ImageUpscaleAE,
)

conf = {
    "model_class": "ImageSingleCoordConvAE",
    "data_path": "./data",
    "epochs": 200,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "in_resolution": 128,
    "in_channels": 3,
    "hidden_dims": [16, 32, 64, 128, 256, 512, 512],
    "hidden_dims_after_first_coordconv": [16, 16, 16],
    # "hidden_dims": [32, 64, 128, 256, 512, 512, 1024],
    "bottleneck_dims": [128, 64, 128],
    # "bottleneck_dims": [256, 128, 256],
    "plot_every_n_epochs": 1,
    "n_plots": 10,
    "log_dir": "./logs",
}

model_class_name = conf["model_class"]
dataset_name = PurePath(conf["data_path"]).name
log_dir = Path(conf["log_dir"]) / model_class_name / dataset_name
log_dir.mkdir(parents=True, exist_ok=True)
run_num = (
    max([int(x.name) for x in log_dir.iterdir()]) + 1 if len(list(log_dir.iterdir())) > 0 else 0
)

log_run_dir = log_dir / str(run_num)
log_run_dir.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if conf["model_class"] == "ImageAE":
    model = ImageAE(
        in_resolution=conf["in_resolution"],
        in_channels=conf["in_channels"],
        hidden_dims=conf["hidden_dims"],
        bottleneck_dims=conf["bottleneck_dims"],
    )
elif conf["model_class"] == "ImageUpscaleAE":
    model = ImageUpscaleAE(
        in_resolution=conf["in_resolution"],
        in_channels=conf["in_channels"],
        hidden_dims=conf["hidden_dims"],
        bottleneck_dims=conf["bottleneck_dims"],
    )
elif conf["model_class"] == "ImageCoordConvAE":
    model = ImageCoordConvAE(
        in_resolution=conf["in_resolution"],
        in_channels=conf["in_channels"],
        hidden_dims=conf["hidden_dims"],
        bottleneck_dims=conf["bottleneck_dims"],
    )
elif conf["model_class"] == "ImageSingleCoordConvAE":
    model = ImageSingleCoordConvAE(
        in_resolution=conf["in_resolution"],
        in_channels=conf["in_channels"],
        hidden_dims=conf["hidden_dims"],
        bottleneck_dims=conf["bottleneck_dims"],
        hidden_dims_after_first_coordconv=conf["hidden_dims_after_first_coordconv"],
    )
else:
    raise ValueError("Unknown model class")

model.to(device)

train_dataset = CausalImageDataset(load_data_parts(conf["data_path"], "train"))
train_loader = DataLoader(
    train_dataset,
    batch_size=conf["batch_size"],
    shuffle=True,
    pin_memory=True,
)

# test_dataset = CausalImageDataset(load_data_parts(conf["data_path"], "test"))
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=conf["batch_size"],
#     shuffle=False,
#     pin_memory=True,
# )

optimizer = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"])

print(f"Starting training...")
for epoch in range(conf["epochs"]):
    model.train()
    for x, *_ in train_loader:
        x = x.to(device)
        x_hat = model(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % conf["plot_every_n_epochs"] == 0:
        fig, axes = plt.subplots(ncols=2, nrows=conf["n_plots"], figsize=(6, 2.5 * conf["n_plots"]))
        for i in range(conf["n_plots"]):
            axes[i, 0].imshow(x[i].detach().cpu().permute(1, 2, 0))
            axes[i, 1].imshow(x_hat[i].detach().cpu().permute(1, 2, 0))
        fig.savefig(log_run_dir / f"epoch_{epoch}.png")
        plt.close(fig)

    print(f"Epoch {epoch}: {loss.item()}")
