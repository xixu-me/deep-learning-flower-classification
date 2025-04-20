import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms

# Import from project files
from model import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
)
from my_dataset import MyDataSet
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from utils import read_split_data


def load_model(model_name, num_classes, device):
    """Load a specific EfficientNet model with its best weights"""
    model_functions = {
        "B0": efficientnet_b0,
        "B1": efficientnet_b1,
        "B2": efficientnet_b2,
        "B3": efficientnet_b3,
        "B4": efficientnet_b4,
        "B5": efficientnet_b5,
    }

    # Create the model
    model = model_functions[model_name](num_classes=num_classes).to(device)

    # Load weights
    weights_path = f"./weights/{model_name}/best_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: No weights found at {weights_path}")

    return model


def evaluate_model(model, data_loader, device, class_names):
    """Evaluate model performance with multiple metrics"""
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            # Record batch inference time
            inference_times.append(end_time - start_time)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays for sklearn metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    results = {
        "accuracy": np.mean(all_preds == all_labels),
        "precision_macro": precision_score(all_labels, all_preds, average="macro"),
        "recall_macro": recall_score(all_labels, all_preds, average="macro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "precision_weighted": precision_score(
            all_labels, all_preds, average="weighted"
        ),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "avg_inference_time": np.mean(inference_times),
        "total_inference_time": np.sum(inference_times),
        "images_per_second": len(all_labels) / np.sum(inference_times),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }

    # Per-class metrics
    class_report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    # Add per-class metrics to results
    for class_name in class_names:
        results[f"precision_{class_name}"] = class_report[class_name]["precision"]
        results[f"recall_{class_name}"] = class_report[class_name]["recall"]
        results[f"f1_{class_name}"] = class_report[class_name]["f1-score"]

    return results


def plot_confusion_matrix(cm, class_names, model_name, save_dir):
    """Plot confusion matrix for a model"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - EfficientNet-{model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/confusion_matrix_{model_name}.png")
    plt.close()


def plot_metrics_comparison(metrics_df, save_dir):
    """Plot comparison of metrics across models"""
    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Selected metrics to plot
    metrics_to_plot = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "images_per_second",
        "avg_inference_time",
    ]

    # Plot each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df["Model"], metrics_df[metric])
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xlabel("Model")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_comparison.png")
        plt.close()

    # Create radar chart for comparing models
    metrics_for_radar = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    # Normalize data for radar chart
    radar_data = metrics_df[metrics_for_radar].values
    radar_data_normalized = radar_data / radar_data.max(axis=0)

    # Plot radar chart
    plt.figure(figsize=(12, 10))
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for i, model in enumerate(metrics_df["Model"]):
        values = radar_data_normalized[i]
        values = np.concatenate((values, [values[0]]))
        plt.polar(angles, values, marker="o", label=f"EfficientNet-{model}")

    plt.xticks(angles[:-1], metrics_for_radar)
    plt.title("Model Performance Comparison (Normalized)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/radar_chart_comparison.png")
    plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class names
    with open("class_indices.json") as f:
        class_mapping = json.load(f)
        class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]

    # Load dataset
    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    # Dictionary to store results for all models
    all_results = []

    # Models to evaluate
    models_to_evaluate = ["B0", "B1", "B2", "B3", "B4", "B5"]

    # Create results directory
    results_dir = "./evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for model_name in models_to_evaluate:
        print(f"\n{'='*50}")
        print(f"Evaluating EfficientNet-{model_name}")
        print(f"{'='*50}")

        # Set up data transformations for this model
        img_size = {
            "B0": 224,
            "B1": 240,
            "B2": 260,
            "B3": 300,
            "B4": 380,
            "B5": 456,
        }

        data_transform = transforms.Compose(
            [
                transforms.Resize(img_size[model_name]),
                transforms.CenterCrop(img_size[model_name]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Create validation dataset
        val_dataset = MyDataSet(
            images_path=val_images_path,
            images_class=val_images_label,
            transform=data_transform,
        )

        # Create validation dataloader
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(
                [os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8]
            ),
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
        )

        # Load model
        model = load_model(model_name, args.num_classes, device)

        # Evaluate model
        results = evaluate_model(model, val_loader, device, class_names)

        # Add model name to results
        results["model"] = f"EfficientNet-{model_name}"

        # Print results summary
        print(f"\nResults for EfficientNet-{model_name}:")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Precision (macro): {results['precision_macro']*100:.2f}%")
        print(f"Recall (macro): {results['recall_macro']*100:.2f}%")
        print(f"F1 Score (macro): {results['f1_macro']*100:.2f}%")
        print(
            f"Average inference time: {results['avg_inference_time']*1000:.2f} ms per batch"
        )
        print(f"Inference speed: {results['images_per_second']:.2f} images/second")

        # Plot confusion matrix
        plot_confusion_matrix(
            results["confusion_matrix"],
            class_names,
            model_name,
            os.path.join(results_dir, "confusion_matrices"),
        )

        # Store results excluding confusion matrix
        results_for_df = {k: v for k, v in results.items() if k != "confusion_matrix"}
        all_results.append(results_for_df)

    # Create DataFrame from all results
    results_df = pd.DataFrame(all_results)

    # Prepare DataFrame for plotting
    plot_df = results_df.rename(columns={"model": "Model"})
    plot_df["Model"] = plot_df["Model"].str.replace("EfficientNet-", "")

    # Plot comparison metrics
    plot_metrics_comparison(plot_df, os.path.join(results_dir, "comparisons"))

    # Save results to CSV
    results_df.to_csv(os.path.join(results_dir, "evaluation_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(results_dir, 'evaluation_results.csv')}")

    # Display comparison table
    comparison_cols = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "avg_inference_time",
        "images_per_second",
    ]

    comparison_df = results_df[comparison_cols].sort_values("accuracy", ascending=False)
    print("\nModel Performance Comparison (sorted by accuracy):")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet models")
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"..\\data\\flower_photos",
        help="dataset root path",
    )
    parser.add_argument("--num_classes", type=int, default=5, help="number of classes")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="batch size for evaluation"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    args = parser.parse_args()
    main(args)
