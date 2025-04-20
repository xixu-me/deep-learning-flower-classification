import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from my_dataset import MyDataSet
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import read_split_data
from vit_model import (
    vit_base_patch16_224_in21k,
    vit_base_patch32_224_in21k,
    vit_large_patch16_224_in21k,
    vit_large_patch32_224_in21k,
)


def load_model(model_name, num_classes, device, weights_path):
    """Load a model based on its name."""
    model_mapping = {
        "vit_base_patch16_224_in21k": vit_base_patch16_224_in21k,
        "vit_base_patch32_224_in21k": vit_base_patch32_224_in21k,
        "vit_large_patch16_224_in21k": vit_large_patch16_224_in21k,
        "vit_large_patch32_224_in21k": vit_large_patch32_224_in21k,
    }

    # Create model
    model = model_mapping[model_name](num_classes=num_classes, has_logits=False)

    # Load weights
    assert os.path.exists(
        weights_path
    ), f"Weights file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.to(device)
    model.eval()

    return model


def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate a model and compute metrics."""
    true_labels = []
    predictions = []

    # For per-class metrics
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

            correct = (preds == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision_macro": precision_score(true_labels, predictions, average="macro"),
        "precision_weighted": precision_score(
            true_labels, predictions, average="weighted"
        ),
        "recall_macro": recall_score(true_labels, predictions, average="macro"),
        "recall_weighted": recall_score(true_labels, predictions, average="weighted"),
        "f1_macro": f1_score(true_labels, predictions, average="macro"),
        "f1_weighted": f1_score(true_labels, predictions, average="weighted"),
        "confusion_matrix": confusion_matrix(true_labels, predictions),
    }

    # Calculate per-class accuracy
    for i in range(num_classes):
        if class_total[i] > 0:
            metrics[f"class_{i}_accuracy"] = class_correct[i] / class_total[i]
        else:
            metrics[f"class_{i}_accuracy"] = 0.0

    return metrics


def visualize_metrics(metrics_dict, class_indict, output_dir):
    """Generate visualizations comparing model performance."""
    os.makedirs(output_dir, exist_ok=True)
    model_names = list(metrics_dict.keys())
    short_names = [name.split("_")[0:3] for name in model_names]
    short_names = [f"{parts[0]} {parts[1]}\n{parts[2]}" for parts in short_names]

    # 1. Overall metrics comparison
    plt.figure(figsize=(14, 8))
    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[model][metric] for model in model_names]
        plt.bar(x + i * width - 0.3, values, width, label=metric_names[i])

    plt.xlabel("Model Architecture")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(x, short_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_metrics.png"))
    plt.close()

    # 2. Per-class accuracy comparison
    plt.figure(figsize=(14, 8))
    num_classes = len(class_indict)

    for i, model_name in enumerate(model_names):
        class_acc = [
            metrics_dict[model_name][f"class_{c}_accuracy"] for c in range(num_classes)
        ]
        plt.plot(
            range(num_classes),
            class_acc,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=short_names[i],
        )

    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy Comparison")
    plt.xticks(range(num_classes), [class_indict[str(i)] for i in range(num_classes)])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"))
    plt.close()

    # 3. Radar chart for comprehensive comparison
    metrics_for_radar = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Create a figure with multiple radar charts
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Define some colors for different models
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, model_name in enumerate(model_names):
        values = [metrics_dict[model_name][metric] for metric in metrics_for_radar]
        values += values[:1]  # Close the loop

        axes.plot(angles, values, color=colors[i], linewidth=2, label=short_names[i])
        axes.fill(angles, values, color=colors[i], alpha=0.25)

    # Set labels and title
    axes.set_xticks(angles[:-1])
    axes.set_xticklabels(labels)
    axes.set_title("Model Performance Radar Chart")
    axes.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"))
    plt.close()


def create_metrics_table(metrics_dict, output_dir):
    """Create and save metrics data in CSV format."""
    # Define which metrics to include in the table
    metrics_to_include = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Get model names
    model_names = list(metrics_dict.keys())
    short_names = [" ".join(name.split("_")[0:3]) for name in model_names]

    # Prepare CSV file path
    csv_path = os.path.join(output_dir, "metrics_table.csv")

    # Write data to CSV
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row with metric names
        header = ["Model"] + metric_names
        writer.writerow(header)

        # Write data for each model
        for i, model_name in enumerate(model_names):
            row = [short_names[i]]
            row.extend(
                [
                    f"{metrics_dict[model_name][metric]:.4f}"
                    for metric in metrics_to_include
                ]
            )
            writer.writerow(row)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT models")
    parser.add_argument(
        "--data-path", type=str, default="../data/flower_photos", help="Dataset path"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="./weights",
        help="Directory containing model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--device", default="cuda:0", help="Device to use")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    # Data preprocessing
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform,
    )

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    # Load class indices
    with open("class_indices.json", "r") as f:
        class_indict = json.load(f)

    # Models to evaluate
    models_to_evaluate = {
        "vit_base_patch16_224_in21k": os.path.join(
            args.weights_dir, "vit_base_patch16_224_in21k", "best_model.pth"
        ),
        "vit_base_patch32_224_in21k": os.path.join(
            args.weights_dir, "vit_base_patch32_224_in21k", "best_model.pth"
        ),
        "vit_large_patch16_224_in21k": os.path.join(
            args.weights_dir, "vit_large_patch16_224_in21k", "best_model.pth"
        ),
        "vit_large_patch32_224_in21k": os.path.join(
            args.weights_dir, "vit_large_patch32_224_in21k", "best_model.pth"
        ),
    }

    all_metrics = {}

    # Evaluate each model
    for model_name, weights_path in models_to_evaluate.items():
        print(f"\nEvaluating {model_name}...")
        model = load_model(model_name, args.num_classes, device, weights_path)
        metrics = evaluate_model(model, val_loader, device, args.num_classes)
        all_metrics[model_name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall: {metrics['recall_macro']:.4f}")
        print(f"F1 Score: {metrics['f1_macro']:.4f}")

    # Visualize results
    visualize_metrics(all_metrics, class_indict, args.output_dir)

    # Create and save metrics table as CSV instead of image
    table_path = create_metrics_table(all_metrics, args.output_dir)
    print(f"Metrics table saved to {table_path}")

    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
