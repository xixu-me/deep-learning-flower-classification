import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from model import densenet121
from my_dataset import MyDataSet
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms
from tqdm import tqdm
from utils import read_split_data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for evaluation")

    # Data transformation for validation
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load validation dataset
    _, _, val_images_path, val_images_label = read_split_data("../data/flower_photos")

    # Create validation dataset
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform,
    )

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    validate_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    # Load class indices
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), f"File: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = densenet121(num_classes=5).to(device)

    # Load model weights
    weights_path = "./weights/best_model.pth"
    assert os.path.exists(weights_path), f"File: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Evaluation
    model.eval()

    # Lists to store predictions and ground truth
    all_preds = []
    all_labels = []

    # Store per-image predictions and confidences for error analysis
    image_paths = []
    image_preds = []
    image_confidences = []
    image_true_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(validate_loader, desc="Evaluating")):
            outputs = model(images.to(device))

            # Get predictions and confidences
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predict_y = torch.max(probabilities, dim=1)

            # Store predictions and labels
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Store image details for error analysis
            for j in range(len(images)):
                if i * batch_size + j < len(val_images_path):
                    image_paths.append(val_images_path[i * batch_size + j])
                    image_preds.append(predict_y[j].item())
                    image_confidences.append(confidence[j].item())
                    image_true_labels.append(labels[j].item())

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

    # Analyze misclassified images
    misclassified = []
    for path, pred, conf, true_label in zip(
        image_paths, image_preds, image_confidences, image_true_labels
    ):
        if pred != true_label:
            misclassified.append(
                {
                    "path": path,
                    "predicted": class_indict[str(pred)],
                    "true": class_indict[str(true_label)],
                    "confidence": conf,
                }
            )

    # Sort misclassified by confidence (high to low)
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)

    # Print most confident misclassifications
    print("\nTop 5 Most Confident Misclassifications:")
    for i, item in enumerate(misclassified[:5]):
        print(f"{i+1}. Path: {os.path.basename(item['path'])}")
        print(
            f"   True: {item['true']}, Predicted: {item['predicted']}, Confidence: {item['confidence']:.4f}"
        )

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
        misclassified[:10],  # Pass top 10 misclassified for visualization
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
    misclassified=None,
):
    """
    Visualize evaluation metrics and confusion matrix
    """
    plt.figure(figsize=(20, 15))

    # Plot confusion matrix with percentages
    plt.subplot(2, 2, 1)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[class_indict[str(i)] for i in range(len(class_indict))],
        yticklabels=[class_indict[str(i)] for i in range(len(class_indict))],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")

    # Plot absolute confusion matrix
    plt.subplot(2, 2, 2)
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
    plt.title("Confusion Matrix (Counts)")

    # Plot overall metrics
    plt.subplot(2, 2, 3)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [accuracy, precision, recall, f1]
    colors = ["#4CAF50", "#2196F3", "#FFC107", "#F44336"]  # Green, Blue, Yellow, Red

    bars = plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1.0)
    plt.title("Overall Performance Metrics")

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # Plot per-class metrics
    plt.subplot(2, 2, 4)
    x = np.arange(len(class_indict))
    width = 0.25

    fig, ax = plt.gcf(), plt.gca()
    rects1 = ax.bar(
        x - width, class_precision, width, label="Precision", color="#2196F3"
    )
    rects2 = ax.bar(x, class_recall, width, label="Recall", color="#FFC107")
    rects3 = ax.bar(x + width, class_f1, width, label="F1 Score", color="#F44336")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-class Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels([class_indict[str(i)] for i in range(len(class_indict))])
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Save main evaluation visualization
    plt.tight_layout()
    plt.savefig("densenet_evaluation_metrics.png")

    # If we have misclassified examples, create an additional visualization
    if misclassified:
        # Create a radar chart for performance metrics
        plt.figure(figsize=(15, 10))

        # Create a radar chart for per-class F1 scores
        plt.subplot(1, 2, 1, polar=True)
        categories = [class_indict[str(i)] for i in range(len(class_indict))]
        N = len(categories)

        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Add metrics
        metrics_data = [
            class_precision.tolist() + [class_precision[0]],  # Close the loop
            class_recall.tolist() + [class_recall[0]],  # Close the loop
            class_f1.tolist() + [class_f1[0]],  # Close the loop
        ]

        labels = ["Precision", "Recall", "F1 Score"]
        colors = ["#2196F3", "#FFC107", "#F44336"]

        ax = plt.subplot(1, 2, 1, polar=True)

        for i, data in enumerate(metrics_data):
            ax.plot(angles, data, linewidth=2, label=labels[i], color=colors[i])
            ax.fill(angles, data, alpha=0.1, color=colors[i])

        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks(
            [0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray"
        )
        plt.ylim(0, 1)
        plt.title("Class Performance Metrics")
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        # Create error analysis pie chart
        plt.subplot(1, 2, 2)

        # Count misclassifications by class
        true_class_counts = {}
        for item in misclassified:
            true_class = item["true"]
            if true_class in true_class_counts:
                true_class_counts[true_class] += 1
            else:
                true_class_counts[true_class] = 1

        # Create pie chart of misclassified true classes
        labels = list(true_class_counts.keys())
        sizes = list(true_class_counts.values())

        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            shadow=True,
            explode=[0.05] * len(labels),
            colors=plt.cm.tab10.colors[: len(labels)],
        )
        plt.axis("equal")
        plt.title("Distribution of Misclassified Images by True Class")

        plt.tight_layout()
        plt.savefig("densenet_error_analysis.png")

    plt.show()


if __name__ == "__main__":
    main()
