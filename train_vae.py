import json
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from dataset import CausalImageDataset, load_data_parts
from experiment_utils import (
    cyclical_annealing_scheduler,
    exponential_scheduler,
    linear_scheduler,
)
from spatialvaes.vaes import ImageVAE, SpatialBroadcastVAE

conf = {
    "warmstart": False,
    "checkpoint_path": None,
    "model_class": "SpatialBroadcastVAE",
    "num_coordinate_channel_pairs": 3,
    "data_path": "./data",
    "pretrain_images": 8,
    "pretrain_epochs": 0,
    "epochs": 400,
    "posttrain_epochs": 0,
    "posttrain_hard_examples": 1000,
    # "vae_beta_schedule": {
    #     "type": "constant_cyclical_annealing",
    #     "initial_constant_epochs": 200,
    #     "num_cycles": 5,
    #     "decay_epochs": 1000,
    #     "initial": 0.0,
    #     "final": 1.0,
    #     "cycle_increase_proportion": 0.5,
    # },
    # "in_resolution": 128,
    "learning_rate": 1e-4,
    "vae_beta_schedule": {
        "type": "constant_linear_constant",
        "initial_constant_epochs": 100,
        "decay_epochs": 250,
        "initial": 0.1,
        "final": 4.0,
    },
    "batch_size": 32,
    "in_resolution": 64,
    "in_channels": 3,
    "latent_dim": 16,
    "hidden_dims_encoder": [16, 32, 64, 128, 128, 256],
    "kernel_size_encoder": 3,
    "hidden_dims_decoder": [128, 64, 32, 16, 8],
    "kernel_size_decoder": 3,
    "fc_dims": [256],
    "plot_every_n_epochs": 1,
    "plot_test_every_n_epochs": 10,
    "sample_every_n_epochs": 10,
    "evaluate_every_n_epochs": 10,
    "save_every_n_epochs": 10,
    "n_plots": 8,
    "log_dir": "./logs",
}


def create_experiment_dir(conf):
    model_class_name = conf["model_class"]
    dataset_name = PurePath(conf["data_path"]).name
    if conf["warmstart"]:
        checkpoint_path = Path(conf["checkpoint_path"])
        original_run_dir = Path(conf["checkpoint_path"]).parent
        log_dir = Path(f"{str(original_run_dir)}_{checkpoint_path.name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        run_num = (
            max([int(x.name) for x in log_dir.iterdir() if x.is_dir() and x.name.isdigit()]) + 1
            if len(list(log_dir.iterdir())) > 0
            else 0
        )
        log_run_dir = log_dir / str(run_num)
        log_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path(conf["log_dir"]) / model_class_name / dataset_name
        log_dir.mkdir(parents=True, exist_ok=True)
        run_num = (
            max([int(x.name) for x in log_dir.iterdir() if x.is_dir() and x.name.isdigit()]) + 1
            if len(list(log_dir.iterdir())) > 0
            else 0
        )
        log_run_dir = log_dir / str(run_num)
        log_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logging to {log_run_dir}")
    return log_run_dir


def create_model(conf):
    if conf["model_class"] == "ImageVAE":
        model = ImageVAE.from_config(conf)
    elif conf["model_class"] == "SpatialBroadcastVAE":
        model = SpatialBroadcastVAE.from_config(conf)
    else:
        raise ValueError("Unknown model class")
    return model


def train(conf, model, train_dataset, train_loader, test_loader, log_run_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"])

    total_epochs = conf["pretrain_epochs"] + conf["epochs"] + conf["posttrain_epochs"]

    epoch = 0
    phases = []

    if conf["pretrain_epochs"] > 0:
        phases.append(("pretraining", conf["pretrain_epochs"]))
    if conf["epochs"] > 0:
        phases.append(("training", conf["epochs"]))
    if conf["posttrain_epochs"] > 0:
        phases.append(("posttraining", conf["posttrain_epochs"]))

    for phase, num_epochs in phases:
        print(f"Starting {phase} phase for {num_epochs} epochs")

        if phase == "pretraining":
            pretrain_indices = torch.randperm(len(train_dataset))[: conf["pretrain_images"]]
            pretrain_dataset = Subset(train_dataset, pretrain_indices)
            loader = DataLoader(
                pretrain_dataset,
                batch_size=conf["batch_size"],
                shuffle=True,
                pin_memory=True,
            )
            print(f"Pretraining on {len(pretrain_dataset)} images")
        elif phase == "posttraining":
            # identify hard examples
            model.eval()
            with torch.no_grad():
                hard_examples = []
                for x, *_ in train_loader:
                    x = x.to(device)
                    x_hat, kl_divergence = model(x)

                    # Only use reconstruction loss
                    mse_loss = F.mse_loss(x_hat, x, reduction="none").view(x.shape[0], -1)
                    mse_loss = mse_loss.sum(dim=-1)

                    hard_examples.extend(zip(mse_loss, x))
                hard_examples.sort(key=lambda x: x[0], reverse=True)
                hard_examples = torch.stack(
                    [x for loss, x in hard_examples[: conf["posttrain_hard_examples"]]]
                )

                # create dataset
                hard_examples_dataset = TensorDataset(
                    hard_examples, torch.zeros(len(hard_examples))
                )
                loader = DataLoader(
                    hard_examples_dataset,
                    batch_size=conf["batch_size"],
                    shuffle=True,
                    # Don't pin memery because data is already on GPU
                    pin_memory=False,
                )
                print(f"Posttraining on {len(hard_examples_dataset)} hard examples")
        else:
            loader = train_loader

        for _ in range(num_epochs):
            model.train()

            if conf["vae_beta_schedule"]["type"] == "constant_exponential_constant":
                beta = exponential_scheduler(
                    epoch - conf["vae_beta_schedule"]["initial_constant_epochs"],
                    conf["vae_beta_schedule"]["decay_epochs"],
                    conf["vae_beta_schedule"]["initial"],
                    conf["vae_beta_schedule"]["final"],
                )
            elif conf["vae_beta_schedule"]["type"] == "constant_linear_constant":
                beta = linear_scheduler(
                    epoch - conf["vae_beta_schedule"]["initial_constant_epochs"],
                    conf["vae_beta_schedule"]["decay_epochs"],
                    conf["vae_beta_schedule"]["initial"],
                    conf["vae_beta_schedule"]["final"],
                )
            elif conf["vae_beta_schedule"]["type"] == "cyclical_annealing":
                beta = cyclical_annealing_scheduler(
                    epoch,
                    conf["vae_beta_schedule"]["decay_epochs"],
                    conf["vae_beta_schedule"]["initial"],
                    conf["vae_beta_schedule"]["final"],
                    conf["vae_beta_schedule"]["num_cycles"],
                    conf["vae_beta_schedule"]["cycle_increase_proportion"],
                )
            elif conf["vae_beta_schedule"]["type"] == "constant_cyclical_annealing":
                beta = cyclical_annealing_scheduler(
                    epoch - conf["vae_beta_schedule"]["initial_constant_epochs"],
                    conf["vae_beta_schedule"]["decay_epochs"],
                    conf["vae_beta_schedule"]["initial"],
                    conf["vae_beta_schedule"]["final"],
                    conf["vae_beta_schedule"]["num_cycles"],
                    conf["vae_beta_schedule"]["cycle_increase_proportion"],
                )
            else:
                beta = 1.0
            train_loss = 0
            train_mse_loss = 0
            train_reg_loss = 0
            for x, *_ in loader:
                x = x.to(device)
                x_hat, kl_divergence = model(x)

                # TODO: take mean of all pixels instead of sum per image
                mse_loss = F.mse_loss(x_hat, x, reduction="none").view(x.shape[0], -1)
                mse_loss = mse_loss.sum(dim=-1).mean()

                reg_loss = kl_divergence.mean() * beta
                loss = mse_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mse_loss += mse_loss.item()
                train_reg_loss += reg_loss.item()

            train_loss /= len(loader)
            train_mse_loss /= len(loader)
            train_reg_loss /= len(loader)
            print(
                f"Epoch {epoch}: {train_loss} (train loss), {train_mse_loss} (mse loss), {train_reg_loss} (reg loss), {beta} (beta)"
            )

            if epoch % conf["plot_every_n_epochs"] == 0 or epoch == total_epochs - 1:
                fig, axes = plt.subplots(
                    ncols=2, nrows=conf["n_plots"], figsize=(6, 2.5 * conf["n_plots"])
                )
                for i in range(conf["n_plots"]):
                    axes[i, 0].imshow(x[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[i, 1].imshow(x_hat[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
                fig.savefig(log_run_dir / f"epoch_{epoch}.png")
                plt.close(fig)

            if epoch % conf["evaluate_every_n_epochs"] == 0 or epoch == total_epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_loss = 0
                    for x, *_ in test_loader:
                        x = x.to(device)
                        x_hat, kl_divergence = model(x)
                        mse_loss = F.mse_loss(x_hat, x, reduction="none").view(x.shape[0], -1)
                        mse_loss = mse_loss.sum(dim=-1).mean()

                        reg_loss = kl_divergence.mean() * beta
                        loss = mse_loss + reg_loss

                        test_loss += loss.item()
                    test_loss /= len(test_loader)
                    print(f"Epoch {epoch}: {test_loss} (test loss)")

            if epoch % conf["plot_test_every_n_epochs"] == 0 or epoch == total_epochs - 1:
                model.eval()
                with torch.no_grad():
                    x_test, *_ = next(iter(test_loader))
                    x_test = x_test.to(device)
                    x_test_hat, _ = model(x_test)
                    fig, axes = plt.subplots(
                        ncols=2,
                        nrows=conf["n_plots"],
                        figsize=(6, 2.5 * conf["n_plots"]),
                    )
                    for i in range(conf["n_plots"]):
                        axes[i, 0].imshow(x_test[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
                        axes[i, 1].imshow(x_test_hat[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
                    fig.savefig(log_run_dir / f"epoch_{epoch}_test.png")
                    plt.close(fig)

            if epoch % conf["sample_every_n_epochs"] == 0 or epoch == total_epochs - 1:
                # sample VAE
                model.eval()
                with torch.no_grad():
                    z_sample = torch.randn(conf["n_plots"], conf["latent_dim"], 1, 1).to(device)
                    x_sample_hat = model.decoder(z_sample)
                    fig, axes = plt.subplots(
                        ncols=1,
                        nrows=conf["n_plots"],
                        figsize=(2, 2.5 * conf["n_plots"]),
                    )
                    for i in range(conf["n_plots"]):
                        axes[i].imshow(x_sample_hat[i].detach().cpu().permute(1, 2, 0).clamp(0, 1))
                    fig.savefig(log_run_dir / f"epoch_{epoch}_sample.png")
                    plt.close(fig)

            if epoch % conf["save_every_n_epochs"] == 0 or epoch == total_epochs - 1:
                torch.save(model.state_dict(), log_run_dir / f"epoch_{epoch}.pt")

            epoch += 1


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    model = create_model(conf)
    if conf["warmstart"]:
        print(f"Warmstarting from {conf['checkpoint_path']}")
        model.load_state_dict(torch.load(conf["checkpoint_path"], map_location=torch.device("cpu")))
    model.to(device)

    log_run_dir = create_experiment_dir(conf)

    # Save config
    with open(log_run_dir / "config.json", "w") as f:
        json.dump(conf, f)

    train_dataset = CausalImageDataset(
        load_data_parts(conf["data_path"], "train"),
        size=(conf["in_resolution"], conf["in_resolution"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    test_dataset = CausalImageDataset(
        load_data_parts(conf["data_path"], "test"),
        size=(conf["in_resolution"], conf["in_resolution"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        pin_memory=True,
    )
    train(conf, model, train_dataset, train_loader, test_loader, log_run_dir)
