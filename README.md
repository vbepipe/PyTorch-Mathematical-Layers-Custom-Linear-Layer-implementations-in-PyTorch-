# PyTorch Mathematical Layers (Custom Linear Layer implementations in PyTorch)
A comprehensive collection of mathematically-inspired custom linear layer implementations in PyTorch, featuring 7 different transformation types with a built-in comparison framework.

## 🚀 Features

- **7 Custom Layer Types**: Standard, Quadratic, Trigonometric, Exponential, Polynomial, Gaussian RBF, and Fourier-inspired transformations
- **Comparative Analysis**: Built-in experimental framework to compare layer performance
- **Synthetic Dataset Generation**: Automated creation of structured classification datasets
- **Professional Architecture**: Clean, well-documented, and extensible codebase
- **Comprehensive Evaluation**: Detailed performance metrics and statistical analysis

## 📦 Custom Layer Types

| Layer Type | Mathematical Transformation | Use Case |
|------------|----------------------------|----------|
| **Standard** | `output = input @ weight.T + bias` | Baseline comparison |
| **Quadratic** | `output = input @ (weight²).T + bias²` | Quadratic relationships |
| **Trigonometric** | `output = sin(input @ weight.T) + cos(bias)` | Periodic patterns |
| **Exponential** | `output = exp(α × (input @ weight.T)) × β + bias` | Growth/decay modeling |
| **Polynomial** | `output = a×z³ + b×z² + c×z + bias` | Complex polynomial fits |
| **Gaussian** | `output = exp(-‖x - w‖² / σ²) + bias` | Local smooth relationships |
| **Fourier** | `output = A×cos(ω×(input @ weight.T) + φ) + bias` | Oscillatory behavior |


# How to Run PyTorch Custom Linear Layers

This guide provides step-by-step instructions to run the PyTorch Custom Linear Layers comparison experiment.

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch 1.9.0 or higher

## Method 1: Clone and Run (Recommended)

### Step 1: Clone the Repository

`git clone [[https://github.com/yourusername/PyTorch-Custom-Linear-Layers.git](https://github.com/vbepipe/PyTorch-Mathematical-Layers-Custom-Linear-Layer-implementations-in-PyTorch-.git)]`

`cd PyTorch-Mathematical-Layers-Custom-Linear-Layer-implementations-in-PyTorch-`

### Step 2: Create Virtual Environment (Optional but Recommended)

**Create virtual environment**

`python -m venv venv`

**Activate virtual environment**

**On Windows:**

`venv\Scripts\activate`

**On macOS/Linux:**

`source venv/bin/activate`

### Step 3: Install Dependencies

`pip install -r requirements.txt`

### Step 4: Run Code

`python custom_linear_layers.py`


## 🚀 Quick Start

### Run the Complete Experiment
This will:
1. Generate a synthetic dataset (1,000 samples, 10 features, 10 classes)
2. Train models with each custom layer type
3. Compare performance across all layers
4. Display comprehensive results

## 🎯 Performance Insights

The experimental framework provides:
- **Accuracy Comparison**: See which layers work best for your data type
- **Loss Analysis**: Understand convergence behavior
- **Statistical Summary**: Get variance and range statistics
- **Ranking System**: Quick identification of top performers

Typical results show:
- **Polynomial layers** often excel at complex, non-linear relationships
- **Fourier layers** work well with periodic or oscillatory data
- **Gaussian layers** are effective for locally-smooth decision boundaries
- **Standard layers** provide reliable baseline performance

## 🔬 Research Applications

This toolkit is valuable for:
- **Neural Architecture Search**: Exploring novel layer designs
- **Mathematical Modeling**: Incorporating domain-specific transformations
- **Educational Purposes**: Understanding different mathematical transformations
- **Baseline Comparisons**: Benchmarking against standard linear layers

## 🤝 Contributing

Contributions are welcome! To add a new custom layer:

1. Implement your layer class inheriting from `nn.Module`
2. Add it to the `custom_layer_map` in `CustomLayerNetwork`
3. Update the documentation and examples
4. Submit a pull request

## 📚 Citation

If you use this code in your research, please cite:

@software{PyTorch-Mathematical-Layers-Custom-Linear-Layer-implementations-in-PyTorch-,

title={PyTorch Mathematical Layers (Custom Linear Layer implementations in PyTorch)},

author={Vinayak Patel},

year={2025},

url={[https://github.com/yourusername/PyTorch-Custom-Linear-Layers](https://github.com/vbepipe/PyTorch-Mathematical-Layers-Custom-Linear-Layer-implementations-in-PyTorch-)} }


## 🔗 Related Work

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [Neural Network Design Patterns](https://pytorch.org/tutorials/)
- [Custom Layer Implementation Guide](https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html)

---

**⭐ Star this repository if you find it useful!**
