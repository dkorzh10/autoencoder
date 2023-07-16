import os
from typing import List

from tqdm import tqdm

import torch
import torch.nn as nn

from src.models import Autoencoder
from src.utils import PATH_TO_SAVE_MODELS


def train(
    train_loader,
    val_loader,
    model,
    device,
    criterion,
    optimizer,
    scheduler,
    c_hid,
    latent_dim,
    epochs=10,
    loss_type="mse",
):
    model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        min_val_loss = 1e20
        epoch_correct = 0
        for batch in train_loader:
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            #             prediction, z = model(image)
            prediction = model(image)

            if loss_type == "mse":
                loss = criterion(prediction, image)
            #                 loss2 = criterion2(z, label)
            #                 loss = loss1 + loss2
            else:
                loss = criterion(prediction, label)
                corr = (prediction.argmax(dim=1) == label).to(torch.int)
                corr = corr.cpu().sum().item()
                epoch_correct += corr

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mean_epoch_loss = epoch_loss / len(train_loader.dataset)
        val_loss = test(val_loader, model, device, criterion, loss_type)

        scheduler.step(val_loss)
        print(mean_epoch_loss, val_loss)
        if loss_type != "mse":
            acc = epoch_correct / len(train_loader.dataset)
            print(f"Train accuracy: {acc}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss

            name = f"loss_type_{loss_type}_c_hid_{c_hid} \
                _latent_dim_{latent_dim}_best_model.pth"
            name = os.path.join(PATH_TO_SAVE_MODELS, name)
            torch.save(model.state_dict(), name)
    return model, mean_epoch_loss


def test(test_loader, model, device, criterion, loss_type):
    model.eval()
    model.to(device)
    test_loss = 0.0
    epoch_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            prediction = model(image)
            if loss_type == "mse":
                loss = criterion(prediction, image)
            else:
                loss = criterion(prediction, label)
                corr = (prediction.argmax(dim=1) == label).to(torch.int)
                corr = corr.cpu().sum().item()
                epoch_correct += corr
            test_loss += loss.item()
    if loss_type != "mse":
        acc = epoch_correct / len(test_loader.dataset)
        print(f"Val/Test accuracy: {acc}")
        return acc

    mean_test_loss = test_loss / len(test_loader.dataset)
    return mean_test_loss


def wrapper(
    train_loader, val_loader,
    latent_dimenshions: List, hidden_channels: List, params
):
    epochs = params["epochs"]
    device = params["device"]
    loss_type = params["loss_type"]
    total_dict = {}
    for c_hid in hidden_channels:
        model_dict = {}
        for latent_dim in latent_dimenshions:
            model = Autoencoder(c_hid, latent_dim)
            criterion = nn.MSELoss(reduction="mean")
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3, min_lr=1e-6
            )

            model, mse = train(
                train_loader,
                val_loader,
                model,
                device,
                criterion,
                optimizer,
                scheduler,
                c_hid,
                latent_dim,
                epochs,
                loss_type,
            )
            model_dict[latent_dim] = {"model": model, "result": mse}
        total_dict[c_hid] = model_dict
    return total_dict
