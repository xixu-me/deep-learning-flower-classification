import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from model import GoogLeNet
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import datasets, transforms
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for evaluation")

    # Data transformation for validation
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load validation dataset
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, "data", "flower_data")
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"), transform=data_transform
    )
    val_num = len(validate_dataset)

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    # Load class indices
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), f"File: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    # Load model weights
    weights_path = "./googleNet.pth"
    assert os.path.exists(weights_path), f"File: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    # Evaluation
    model.eval()

    # Lists to store predictions and ground truth
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for val_data in tqdm(validate_loader, desc="Evaluating"):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))

            # Get predictions
            predict_y = torch.max(outputs, dim=1)[1]

            # Store predictions and labels
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(val_labels.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Calculate per-class metrics
    class_precision = precision_score(all_labels, all_preds, average=None)
    class_recall = recall_score(all_labels, all_preds, average=None)
    class_f1 = f1_score(all_labels, all_preds, average=None)

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print per-class metrics
    print("\nPer-class Metrics:")
    for i in range(len(class_indict)):
        class_name = class_indict[str(i)]
        print(f"Class: {class_name}")
        print(f"  Precision: {class_precision[i]:.4f}")
        print(f"  Recall: {class_recall[i]:.4f}")
        print(f"  F1 Score: {class_f1[i]:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Visualize results
    visualize_results(
        cm,
        class_indict,
        accuracy,
        precision,
        recall,
        f1,
        class_precision,
        class_recall,
        class_f1,
    )


def visualize_results(
    cm,
    class_indict,
    accuracy,
    precision,
    recall,
    f1,
    class_precision,
    class_recall,
    class_f1,
):
    """
    Visualize evaluation metrics and confusion matrix
    """
    plt.figure(figsize=(16, 12))

    # Plot confusion matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_indict[str(i)] for i in range(len(class_indict))],
        yticklabels=[class_indict[str(i)] for i in range(len(class_indict))],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Plot overall metrics
    plt.subplot(2, 2, 2)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1]
    plt.bar(metrics, values, color=["blue", "green", "orange", "red"])
    plt.ylim(0, 1.0)
    plt.title("Overall Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")

    # Plot per-class precision, recall, and F1 score
    plt.subplot(2, 2, 3)
    x = np.arange(len(class_indict))
    width = 0.25

    plt.bar(x - width, class_precision, width, label="Precision", color="green")
    plt.bar(x, class_recall, width, label="Recall", color="orange")
    plt.bar(x + width, class_f1, width, label="F1 Score", color="red")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Per-class Metrics")
    plt.xticks(x, [class_indict[str(i)] for i in range(len(class_indict))])
    plt.legend()
    plt.ylim(0, 1.0)

    # Plot ROC curve (simplified version for multiclass)
    plt.subplot(2, 2, 4)

    # Create a radar chart for per-class F1 scores
    categories = [class_indict[str(i)] for i in range(len(class_indict))]
    N = len(categories)

    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Add F1 scores
    values = class_f1.tolist()
    values += values[:1]  # Close the loop

    # Draw the plot
    ax = plt.subplot(2, 2, 4, polar=True)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        color="grey",
        size=7,
    )
    plt.ylim(0, 1)

    plt.plot(angles, values, linewidth=1, linestyle="solid")
    plt.fill(angles, values, "b", alpha=0.1)
    plt.title("F1 Score by Class")

    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.show()


if __name__ == "__main__":
    main()
