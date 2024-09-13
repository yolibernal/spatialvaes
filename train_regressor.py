from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CausalImageDataset, load_data_parts
from spatialvaes.layers import CoordConv2d
from spatialvaes.regressors import ImageRegressor

conf = {
    "model_class": "ImageCoordConvRegressor",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "in_resolution": 128,
    "in_channels": 3,
    "hidden_dims": [16, 32, 64, 128, 256, 512, 512],
    "fully_connected_dims": [512, 256, 128, 64],
    "evaluate_every_n_epochs": 1,
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

if conf["model_class"] == "ImageRegressor":
    model = ImageRegressor(
        in_channels=conf["in_channels"],
        hidden_dims=conf["hidden_dims"],
        fully_connected_dims=conf["fully_connected_dims"],
        output_dim=6,
    )
if conf["model_class"] == "ImageCoordConvRegressor":
    model = ImageRegressor(
        in_channels=conf["in_channels"],
        hidden_dims=conf["hidden_dims"],
        fully_connected_dims=conf["fully_connected_dims"],
        output_dim=6,
        conv_class=CoordConv2d,
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
        x = x.to(device)
        label = label.to(device)
        label_hat = model(x)

        loss = torch.nn.functional.mse_loss(label_hat, label)
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
                x = x.to(device)
                label = label.to(device)
                label_hat = model(x)
                loss = torch.nn.functional.mse_loss(label_hat, label)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Epoch {epoch}: test {test_loss}")

            # Accuracy of predicted pixels
            accuracy_threshold = 1
            correct = 0
            total = 0
            for x, z, e, intervention_label, intervention_mask in test_loader:
                if label_type == "z":
                    label = z
                elif label_type == "e":
                    label = e
                else:
                    raise ValueError(f"Unknown label type {label_type}")
                x = x.to(device)
                label = label.to(device)
                label_hat = model(x)
                label_diffs = torch.abs(label - label_hat)
                correct += torch.sum(label_diffs < accuracy_threshold).item()
                total += torch.numel(label_diffs)
            print(f"Accuracy: {correct / total}")

    if epoch % conf["plot_every_n_epochs"] == 0:
        model.eval()
        with torch.no_grad():
            for i, (x, z, e, intervention_label, intervention_mask) in enumerate(test_loader):
                if label_type == "z":
                    label = z
                elif label_type == "e":
                    label = e
                else:
                    raise ValueError(f"Unknown label type {label_type}")
                x = x.to(device)
                label = label.to(device)
                label_hat = model(x)
                break

            ncols = 2
            nrows = np.ceil(conf["n_plots"] / ncols).astype(int)
            fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(6, 2.5 * conf["n_plots"]))
            for i in range(conf["n_plots"]):
                axes.flat[i].imshow(x[i].detach().cpu().permute(1, 2, 0))
                p1 = label[i, :2].detach().cpu().numpy()
                p2 = label[i, 2:4].detach().cpu().numpy()
                p3 = label[i, 4:].detach().cpu().numpy()

                p1_hat = label_hat[i, :2].detach().cpu().numpy()
                p2_hat = label_hat[i, 2:4].detach().cpu().numpy()
                p3_hat = label_hat[i, 4:].detach().cpu().numpy()

                axes.flat[i].scatter(p1[0], p1[1], c="r", s=10)
                axes.flat[i].scatter(p2[0], p2[1], c="r", s=10)
                axes.flat[i].scatter(p3[0], p3[1], c="r", s=10)

                axes.flat[i].scatter(p1_hat[0], p1_hat[1], c="b", s=10)
                axes.flat[i].scatter(p2_hat[0], p2_hat[1], c="b", s=10)
                axes.flat[i].scatter(p3_hat[0], p3_hat[1], c="b", s=10)

            fig.savefig(log_run_dir / f"epoch_{epoch}.png")
            plt.close(fig)
