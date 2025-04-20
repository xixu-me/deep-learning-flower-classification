import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Import all EfficientNet models instead of just B0
from model import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)
from my_dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import evaluate, read_split_data, train_one_epoch

# Dictionary mapping model names to their respective functions
model_functions = {
    "B0": efficientnet_b0,
    "B1": efficientnet_b1,
    "B2": efficientnet_b2,
    "B3": efficientnet_b3,
    "B4": efficientnet_b4,
    "B5": efficientnet_b5,
    "B6": efficientnet_b6,
    "B7": efficientnet_b7,
}


def plot_training_process(
    train_losses, val_accuracies, learning_rates, save_dir="./plots"
):
    """
    Plot training metrics and save the figures
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_loss.png")
    plt.close()

    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/validation_accuracy.png")
    plt.close()

    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, label="Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/learning_rate.png")
    plt.close()

    # Combined plot
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(train_losses, "b-", label="Training Loss")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(val_accuracies, "r-", label="Validation Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(learning_rates, "g-", label="Learning Rate")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Learning Rate")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_summary.png")
    plt.close()


def train_model(
    model_name,
    args,
    train_images_path,
    train_images_label,
    val_images_path,
    val_images_label,
):
    """
    Function to train a specific EfficientNet model variant
    """
    print(f"\n{'='*50}")
    print(f"Starting training for EfficientNet-{model_name}")
    print(f"{'='*50}\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create model-specific directory for weights and plots
    model_weights_dir = f"./weights/{model_name}"
    model_plots_dir = f"./plots/{model_name}"
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    if not os.path.exists(model_plots_dir):
        os.makedirs(model_plots_dir)

    # Create a new SummaryWriter for this model
    tb_writer = SummaryWriter(log_dir=f"runs/efficientnet_{model_name.lower()}")

    img_size = {
        "B0": 224,
        "B1": 240,
        "B2": 260,
        "B3": 300,
        "B4": 380,
        "B5": 456,
        "B6": 528,
        "B7": 600,
    }

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size[model_name]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(img_size[model_name]),
                transforms.CenterCrop(img_size[model_name]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create datasets with model-specific transforms
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
    print(f"Using {nw} dataloader workers every process")

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

    # Create model using the appropriate function from the dictionary
    create_model = model_functions[model_name]
    model = create_model(num_classes=args.num_classes).to(device)

    # Model-specific weights path
    weights_path = f"./efficientnet{model_name.lower()}.pth"

    # Load pretrained weights if available
    if args.weights != "" and model_name == "B0":
        # Only use the specified weights for B0
        weights_path = args.weights

    if os.path.exists(weights_path):
        print(f"Loading pretrained weights from: {weights_path}")
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {
            k: v
            for k, v in weights_dict.items()
            if model.state_dict()[k].numel() == v.numel()
        }
        print(model.load_state_dict(load_weights_dict, strict=False))
    else:
        print(f"No pretrained weights found at: {weights_path}, starting from scratch")

    # Handle layer freezing
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print(f"Training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    learning_rates = []

    for epoch in range(args.epochs):
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        acc = evaluate(model=model, data_loader=val_loader, device=device)
        print(f"[epoch {epoch}] accuracy: {round(acc, 3)}")

        train_losses.append(mean_loss)
        val_accuracies.append(acc)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Log metrics
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # Save model weights periodically
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{model_weights_dir}/model-{epoch}.pth")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{model_weights_dir}/best_model.pth")

    # Plot training metrics for this model
    plot_training_process(
        train_losses, val_accuracies, learning_rates, save_dir=model_plots_dir
    )

    print(f"Best accuracy for EfficientNet-{model_name}: {best_acc:.4f}")
    print(f"Training visualizations saved to {model_plots_dir}")

    # Close the tensorboard writer
    tb_writer.close()

    return best_acc


def main(args):
    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # Read dataset only once
    train_images_path, train_images_label, val_images_path, val_images_label = (
        read_split_data(args.data_path)
    )

    # List of all EfficientNet models to train sequentially
    model_variants = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    # Dictionary to store best accuracy for each model
    best_accuracies = {}

    # Train each model sequentially
    for model_name in model_variants:
        best_acc = train_model(
            model_name=model_name,
            args=args,
            train_images_path=train_images_path,
            train_images_label=train_images_label,
            val_images_path=val_images_path,
            val_images_label=val_images_label,
        )
        best_accuracies[model_name] = best_acc

    # Print summary of all models
    print("\n" + "=" * 50)
    print("Training complete for all EfficientNet models!")
    print("Best accuracies for each model:")
    for model_name, acc in best_accuracies.items():
        print(f"EfficientNet-{model_name}: {acc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument("--data-path", type=str, default=r"..\\data\\flower_photos")

    parser.add_argument(
        "--weights",
        type=str,
        default="./efficientnetb0.pth",
        help="initial weights path",
    )
    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)
