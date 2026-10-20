# CIFAR-10 Deep Learning Experiments

A TensorFlow/Keras project that explores and compares several neural-network design choices for image classification on the CIFAR-10 dataset.

The project starts with fully connected baselines, then investigates normalization and activation/initialization choices, and finally builds a custom convolutional neural network (CNN). It also includes a related Horse vs. Not-Horse feature-reuse experiment.

## Project Goals

The main goals of this project are to:

- compare dense neural networks with convolutional neural networks for image classification;
- study the effect of ReLU + He initialization;
- investigate Batch Normalization and its placement;
- experiment with SELU + LeCun Normal initialization;
- visualize convolutional filters and feature maps;
- build and evaluate a custom deep CNN on CIFAR-10;
- reuse learned CNN features for a related binary classification task.

## Dataset

This project uses the **CIFAR-10** dataset.

CIFAR-10 contains 60,000 RGB images with a resolution of **32 × 32** pixels, divided into 10 classes.

The image values are scaled to the `[0, 1]` range before training.

## Experiments

### 1. Fully Connected Baseline

The first model uses a flattened CIFAR-10 image followed by two hidden layers:

```text
Flatten
→ Dense(50, ReLU)
→ Dense(50, ReLU)
→ Dense(10, Softmax)
```

The hidden layers use **ReLU** activation with **He Normal** initialization.

This model serves as a baseline and demonstrates the limitations of flattening images before classification.

### 2. Batch Normalization

Batch Normalization is added to the dense architecture to study its effect on training and validation performance.

The notebook also investigates different placements of Batch Normalization relative to the activation function.

### 3. SELU Network

A self-normalizing dense network is tested using:

```text
SELU activation
+
LeCun Normal initialization
```

This experiment explores an alternative to ReLU-based dense networks and explicit Batch Normalization.

### 4. Convolution Visualization

Simple convolutional filters are applied to images to visualize how convolution transforms an image into feature maps.

This section is intended to build intuition about local feature extraction before constructing the deeper CNN.

### 5. Custom CNN

The main convolutional model contains six convolutional layers:

```text
Input: 32×32×3

Conv2D(32, 3×3)
Conv2D(32, 3×3)
MaxPooling

Conv2D(64, 3×3)
Conv2D(64, 3×3)
MaxPooling

Conv2D(128, 3×3)
Conv2D(128, 3×3)
MaxPooling

GlobalAveragePooling2D
Dense(10, Softmax)
```

The convolutional layers use:

- `3×3` kernels
- ReLU activation
- He Normal initialization
- `same` padding

The model has approximately **288K trainable parameters**.

## Main CNN Result

The custom CNN achieved the following result on the CIFAR-10 test set:

| Metric | Result |
|---|---:|
| Test Accuracy | **77.99%** |
| Test Loss | **0.6863** |
| Training Time | **~16.2 minutes** |

The CNN substantially improves over the simple dense baseline because convolution preserves and exploits spatial structure in the images.

## Horse vs. Not-Horse Experiment

The trained CNN is also reused as a feature extractor for a binary classification task:

```text
Horse     → 1
Not Horse → 0
```

A balanced subset is created, the convolutional feature extractor is frozen, and a new sigmoid classification layer is trained.

This section demonstrates feature reuse in a related classification task.

> Note: the binary-classification section is kept as an experimental extension. Its result is not used as the main reported project metric.

## Project Structure

```text
cifar10-deep-learning-experiments/
│
├── ann.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   ├── best_cnn.keras
│   └── best_cnn_transfer.keras
│
├── histories/
│   ├── model2_history.npz
│   ├── model3_history.npz
│   └── model4_history.npz
│
└── figures/
```

The saved model and history files make it possible to inspect trained models and reproduce plots without retraining every experiment.

## Installation

Clone the repository:

```bash
git clone https://github.com/arminaamini/cifar10-deep-learning-experiments.git
cd cifar10-deep-learning-experiments
```

If a `requirements.txt` file is included, install the dependencies with:

```bash
pip install -r requirements.txt
```

Otherwise, the main dependencies are:

```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

Then open the notebook:

```bash
jupyter notebook ann.ipynb
```

You can also open the project directly in VS Code.

## Main Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## What I Learned

This project helped me study and apply several important deep-learning concepts:

- activation functions and compatible weight initialization;
- Batch Normalization;
- self-normalizing networks;
- dense vs. convolutional architectures;
- convolutional feature extraction;
- pooling and Global Average Pooling;
- early stopping and model checkpointing;
- validation/test separation;
- saved training histories and trained-model reuse.

## Possible Future Improvements

Future extensions could include:

- data augmentation;
- learning-rate scheduling;
- Dropout and other regularization techniques;
- comparison of different pooling strategies;
- confusion matrix and per-class evaluation;
- pretrained architectures and transfer learning;
- additional optimizer and learning-rate experiments.

## Author

Built as a hands-on deep-learning project using TensorFlow/Keras.
