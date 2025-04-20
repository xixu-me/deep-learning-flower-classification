import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import vit_model
from my_dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import evaluate, read_split_data, train_one_epoch


def train_model(
    args,
    model_name,
    model_creator,
    pretrained_weights,
    device,
    train_loader,
    val_loader,
    tb_writer,
    history,
):
    print(f"\n{'='*20} Training {model_name} {'='*20}")

    # Create model
    model = model_creator(num_classes=args.num_classes, has_logits=False).to(device)

    # Load pretrained weights
    if os.path.exists(pretrained_weights):
        print(f"Loading pretrained weights: {pretrained_weights}")
        weights_dict = torch.load(pretrained_weights, map_location=device)
        # Remove classifier weights that don't match (num_classes difference)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(f"Loaded pretrained weights from {pretrained_weights}")
        model.load_state_dict(weights_dict, strict=False)
    else:
        print(f"Warning: Pretrained weights {pretrained_weights} not found!")

    # Freeze layers if needed
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"Training {name}")

    # Set up optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)

    # Set up scheduler
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Training loop
    best_acc = 0.0
    model_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        # Evaluate
        val_loss, val_acc = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch
        )

        # Record metrics
        model_history["train_loss"].append(train_loss)
        model_history["train_acc"].append(train_acc)
        model_history["val_loss"].append(val_loss)
        model_history["val_acc"].append(val_acc)

        # TensorBoard logging
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(f"{model_name}/{tags[0]}", train_loss, epoch)
        tb_writer.add_scalar(f"{model_name}/{tags[1]}", train_acc, epoch)
        tb_writer.add_scalar(f"{model_name}/{tags[2]}", val_loss, epoch)
        tb_writer.add_scalar(f"{model_name}/{tags[3]}", val_acc, epoch)
        tb_writer.add_scalar(
            f"{model_name}/{tags[4]}", optimizer.param_groups[0]["lr"], epoch
        )

        # Save model
        model_dir = f"./weights/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_dir}/model-{epoch}.pth")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{model_dir}/best_model.pth")

    # Store the history for this model
    history[model_name] = model_history
    return history


def plot_training_results(history):
    """Plot the training and validation results for all models"""
    models = list(history.keys())
    epochs = range(1, len(history[models[0]]["train_loss"]) + 1)

    plt.figure(figsize=(20, 15))

    # Plot training & validation loss
    plt.subplot(2, 1, 1)
    for model_name in models:
        plt.plot(
            epochs, history[model_name]["train_loss"], "-o", label=f"{model_name} Train"
        )
        plt.plot(
            epochs, history[model_name]["val_loss"], "-s", label=f"{model_name} Val"
        )

    plt.title("Training and Validation Loss", fontsize=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Plot training & validation accuracy
    plt.subplot(2, 1, 2)
    for model_name in models:
        plt.plot(
            epochs, history[model_name]["train_acc"], "-o", label=f"{model_name} Train"
        )
        plt.plot(
            epochs, history[model_name]["val_acc"], "-s", label=f"{model_name} Val"
        )

    plt.title("Training and Validation Accuracy", fontsize=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./training_results.png", dpi=300)
    plt.show()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = (
        read_split_data(args.data_path)
    )

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    }

    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )

    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader workers every process".format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    # Define models to train and their corresponding pretrained weight files
    models_to_train = {
        "vit_base_patch16_224_in21k": {
            "creator": vit_model.vit_base_patch16_224_in21k,
            "weights": "jx_vit_base_patch16_224_in21k-e5005f0a.pth",
        },
        "vit_base_patch32_224_in21k": {
            "creator": vit_model.vit_base_patch32_224_in21k,
            "weights": "jx_vit_base_patch32_224_in21k-8db57226.pth",
        },
        "vit_large_patch16_224_in21k": {
            "creator": vit_model.vit_large_patch16_224_in21k,
            "weights": "jx_vit_large_patch16_224_in21k-606da67d.pth",
        },
        "vit_large_patch32_224_in21k": {
            "creator": vit_model.vit_large_patch32_224_in21k,
            "weights": "jx_vit_large_patch32_224_in21k-9046d2e7.pth",
        },
    }

    # Dictionary to store training history for all models
    training_history = {}

    # Train each model
    for model_name, model_info in models_to_train.items():
        training_history = train_model(
            args=args,
            model_name=model_name,
            model_creator=model_info["creator"],
            pretrained_weights=model_info["weights"],
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            tb_writer=tb_writer,
            history=training_history,
        )

    # Plot and save the training results
    plot_training_results(training_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument(
        "--epochs", type=int, default=10
    )  # Reduced epochs for testing all models
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)

    parser.add_argument("--data-path", type=str, default="../data/flower_photos")
    parser.add_argument("--model-name", default="", help="create model name")

    parser.add_argument("--weights", type=str, default="", help="initial weights path")

    parser.add_argument("--freeze-layers", type=bool, default=True)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)
