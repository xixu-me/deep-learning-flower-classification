# Deep Learning Flower Classification

This repository contains implementations of various deep learning models for classifying flower images.

## Models Included

The following models are implemented in this repository:

* **DenseNet:** [DenseNet/](DenseNet/)
* **EfficientNet:** [EfficientNet/](EfficientNet/)
* **GoogLeNet:** [GoogLeNet/](GoogLeNet/)
* **Transformer:** [Transformer/](Transformer/)

## Data

The project uses flower image data located in:

* `data/flower_data/`: Processed data for training and validation.
* `data/flower_photos/`: Original flower photos.

*Note: Data and pre-trained model weights are excluded from the repository via the [`.gitignore`](.gitignore) file.*

## Project Structure

Each model resides in its own directory (e.g., `DenseNet/`, `EfficientNet/`). Within each model's directory, you will typically find:

* `model.py`: Defines the model architecture.
* `train.py`: Script for training the model.
* `evaluate.py`: Script for evaluating the trained model.
* `predict.py`: Script for making predictions on new images.
* `my_dataset.py`: Defines the custom dataset loading logic.
* `utils.py`: Contains utility functions.
* `class_indices.json`: Maps class indices to class names.

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/xixu-me/deep-learning-flower-classification.git
    cd deep-learning-flower-classification
    ```

2. **Prepare Data:** Ensure the required datasets are present in the `data/` directory as expected by the scripts.
3. **Install Dependencies:** Install necessary Python libraries (e.g., PyTorch, torchvision, numpy, matplotlib). *Consider adding a `requirements.txt` file.*
4. **Navigate to a Model Directory:**

    ```bash
    cd DenseNet/ # or EfficientNet/, GoogLeNet/, Transformer/
    ```

5. **Train:**

    ```bash
    python train.py # Add necessary arguments
    ```

6. **Evaluate:**

    ```bash
    python evaluate.py # Add necessary arguments
    ```

7. **Predict:**

    ```bash
    python predict.py --image_path <path_to_image> # Add necessary arguments
    ```

*Refer to the specific scripts within each model directory for detailed usage instructions and available arguments.*

## License

Copyright &copy; [Xi Xu](https://xi-xu.me). All rights reserved.

Licensed under the [GPL-3.0](LICENSE) license.
