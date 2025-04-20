import argparse
import math
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import densenet121, load_state_dict
from my_dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import evaluate, read_split_data, train_one_epoch


def download_weights(url, filename):
    """
    Download weights file if it doesn't exist locally
    """
    if not os.path.exists(filename):
        print(f"Downloading weights file from {url}...")
        try:
            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(filename) if os.path.dirname(filename) else ".",
                exist_ok=True,
            )
            # Download the file
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded weights file to {filename}")
            return True
        except Exception as e:
            print(f"Error downloading weights file: {e}")
            return False
    return True


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = (
        read_split_data(args.data_path)
    )

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )

    # 实例化验证数据集
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )

    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
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

    # 如果存在预训练权重则载入
    model = densenet121(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            load_state_dict(model, args.weights)
        else:
            # Try to download the weights file
            weights_url = "https://download.pytorch.org/models/densenet121-a639ec97.pth"
            if download_weights(weights_url, args.weights):
                load_state_dict(model, args.weights)
            else:
                print(
                    f"Warning: Could not download weights file. Training from scratch."
                )

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        pg, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Lists to store metrics for plotting
    train_losses = []
    val_accuracies = []
    learning_rates = []

    best_acc = 0.0
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # validate
        acc = evaluate(model=model, data_loader=val_loader, device=device)

        # Store metrics for plotting
        train_losses.append(mean_loss)
        val_accuracies.append(acc)

        print(
            "[epoch %d] train_loss: %.3f  val_accuracy: %.3f  lr: %.6f"
            % (epoch + 1, mean_loss, acc, current_lr)
        )

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], current_lr, epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        if acc > best_acc:
            best_acc = acc
            # Save the model with the best validation accuracy
            torch.save(model.state_dict(), "./weights/best_model.pth")

    # When training is complete, visualize the training process
    visualize_training(args.epochs, train_losses, val_accuracies, learning_rates)


def visualize_training(epochs, train_losses, val_accuracies, learning_rates):
    """
    Create and save visualizations of the training process
    """
    plt.figure(figsize=(15, 10))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(
        range(1, epochs + 1), train_losses, "b-", marker="o", label="Training Loss"
    )
    plt.title("Training Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(
        range(1, epochs + 1),
        val_accuracies,
        "r-",
        marker="o",
        label="Validation Accuracy",
    )
    plt.title("Validation Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(
        range(1, epochs + 1), learning_rates, "m-", marker="o", label="Learning Rate"
    )
    plt.title("Learning Rate vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.legend()

    # Plot loss vs. accuracy
    plt.subplot(2, 2, 4)
    plt.scatter(
        train_losses,
        val_accuracies,
        c=range(epochs),
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    plt.colorbar(label="Epoch")
    plt.title("Validation Accuracy vs. Training Loss")
    plt.xlabel("Training Loss")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("densenet_training_visualization.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size", type=int, default=4
    )  # 如果cuda报超出显存可以改小一点
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument("--data-path", type=str, default="../data/flower_photos")

    # densenet121 官方权重下载地址
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument(
        "--weights",
        type=str,
        default="densenet121-a639ec97.pth",
        help="initial weights path",
    )
    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)
