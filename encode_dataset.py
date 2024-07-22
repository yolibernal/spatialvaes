import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CausalImageDataset, load_data_parts
from spatialvaes.vaes import ImageVAE, SpatialBroadcastVAE
from train_vae import create_model

EXPERIMENT_PATH = Path("logs/SpatialBroadcastVAE/data/0")

STATE_DICT_PATH = EXPERIMENT_PATH / "epoch_399.pt"
CONFIG_PATH = EXPERIMENT_PATH / "config.json"

OUT_DIR = Path("encoded/data/0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH) as f:
    conf = json.load(f)

model = create_model(conf)
model.load_state_dict(torch.load(STATE_DICT_PATH))
model.to(device)

tags = ["train", "val", "test", "dci_train"]

DATA_PATH = None
if DATA_PATH is None:
    DATA_PATH = Path(conf["data_path"])

for tag in tags:
    dataset = CausalImageDataset(
        load_data_parts(DATA_PATH, tag),
        size=(conf["in_resolution"], conf["in_resolution"]),
        merge_before_after=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        shuffle=False,
        pin_memory=True,
    )
    print(f"Encoding {tag} dataset...")

    z0s = []
    z1s = []
    true_z0s = []
    true_z1s = []
    intervention_labels = []
    interventions = []
    true_e0s = []
    true_e1s = []

    for i, (x0, x1, z0, z1, intervention_label, intervention_mask, e0, e1) in enumerate(loader):
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = z0.to(device)
        z1 = z1.to(device)
        intervention_label = intervention_label.to(device)
        intervention_mask = intervention_mask.to(device)
        e0 = e0.to(device)
        e1 = e1.to(device)

        with torch.no_grad():
            features0 = model.encoder(x0)
            features1 = model.encoder(x1)
            features0 = features0.view(features0.size(0), -1)
            features1 = features1.view(features1.size(0), -1)

            mu0 = model.encoder_mu(features0)
            mu1 = model.encoder_mu(features1)

            z0_hat = mu0
            z1_hat = mu1

            z0s.append(z0_hat)
            z1s.append(z1_hat)
            true_z0s.append(z0)
            true_z1s.append(z1)
            intervention_labels.append(intervention_label)
            interventions.append(intervention_mask)
            true_e0s.append(e0)
            true_e1s.append(e1)

    z0s = torch.cat(z0s, dim=0)
    z1s = torch.cat(z1s, dim=0)
    true_z0s = torch.cat(true_z0s, dim=0)
    true_z1s = torch.cat(true_z1s, dim=0)
    intervention_labels = torch.cat(intervention_labels, dim=0)
    interventions = torch.cat(interventions, dim=0)
    true_e0s = torch.cat(true_e0s, dim=0)
    true_e1s = torch.cat(true_e1s, dim=0)

    data = (
        z0s,
        z1s,
        true_z0s,
        true_z1s,
        intervention_labels,
        interventions,
        true_e0s,
        true_e1s,
    )
    filename = OUT_DIR / f"{tag}_encoded.pt"
    print(f"Storing encoded {tag} data at {filename}")

    torch.save(data, filename)
