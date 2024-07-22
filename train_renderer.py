from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CausalImageDataset, load_data_parts
from spatialvaes.renderer import (
    ImageCoordConvRenderer,
    ImageRenderer,
    ImageSingleCoordConvRenderer,
    ImageUpscaleRenderer,
)

conf = {
    "model_class": "ImageRenderer",
    "data_path": "./data",
    "epochs": 200,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "in_resolution": 128,
    "input_dim": 6,
    "hidden_dims": [2048, 1024, 512, 128, 64, 32, 16],
    "hidden_dims_after_first_coordconv": [16, 16, 16],
    "evaluate_every_n_epochs": 10,
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

if conf["model_class"] == "ImageRenderer":
    model = ImageRenderer(input_dim=conf["input_dim"], hidden_dims=conf["hidden_dims"])
elif conf["model_class"] == "ImageUpscaleRenderer":
    model = ImageUpscaleRenderer(input_dim=conf["input_dim"], hidden_dims=conf["hidden_dims"])
elif conf["model_class"] == "ImageCoordConvRenderer":
    model = ImageCoordConvRenderer(input_dim=conf["input_dim"], hidden_dims=conf["hidden_dims"])
elif conf["model_class"] == "ImageSingleCoordConvRenderer":
    model = ImageSingleCoordConvRenderer(
        input_dim=conf["input_dim"],
        hidden_dims=conf["hidden_dims"],
        hidden_dims_after_first_coordconv=conf["hidden_dims_after_first_coordconv"],
    )
else:
    raise ValueError("Unknown model class")

model.to(device)

label_type = "z"
train_dataset = CausalImageDataset(load_data_parts(conf["data_path"], "train"))
train_loader = DataLoader(
    train_dataset,
    batch_size=conf["batch_size"],
    shuffle=True,
    pin_memory=True,
)

test_dataset = CausalImageDataset(load_data_parts(conf["data_path"], "test"))
test_loader = DataLoader(
    test_dataset,
    batch_size=conf["batch_size"],
    shuffle=False,
    pin_memory=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"])

print(f"Starting training...")
for epoch in range(conf["epochs"]):
    model.train()
    train_loss = 0
    for x, z, e, intervention_label, intervention_mask in train_loader:
        if label_type == "z":
            label = z
        elif label_type == "e":
            label = e
        else:
            raise ValueError(f"Unknown label type {label_type}")
        # filter x where label contains NaNs
        x = x[~torch.isnan(label).any(dim=1)]
        label = label[~torch.isnan(label).any(dim=1)]

        x = x.to(device)
        label = label.to(device)
        x_hat = model(label)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch}: train {train_loss}")

    if epoch % conf["evaluate_every_n_epochs"] == 0:
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for x, z, e, intervention_label, intervention_mask in test_loader:
                if label_type == "z":
                    label = z
                elif label_type == "e":
                    label = e
                else:
                    raise ValueError(f"Unknown label type {label_type}")
                # filter x where label contains NaNs
                x = x[~torch.isnan(label).any(dim=1)]
                label = label[~torch.isnan(label).any(dim=1)]

                x = x.to(device)
                label = label.to(device)
                x_hat = model(label)
                loss = torch.nn.functional.mse_loss(x_hat, x)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Epoch {epoch}: test {test_loss}")

    if epoch % conf["plot_every_n_epochs"] == 0:
        fig, axes = plt.subplots(ncols=2, nrows=conf["n_plots"], figsize=(6, 2.5 * conf["n_plots"]))
        for i in range(conf["n_plots"]):
            axes[i, 0].imshow(x[i].detach().cpu().permute(1, 2, 0))
            axes[i, 1].imshow(x_hat[i].detach().cpu().permute(1, 2, 0))
        fig.savefig(log_run_dir / f"epoch_{epoch}.png")
        plt.close(fig)
