"""
Custom Linear Layer Implementations in PyTorch
==============================================

This module implements various mathematical transformations as custom linear layers
that can be used as drop-in replacements for standard PyTorch linear layers.

Author: [Your Name]
Date: [Current Date]
License: MIT

Features:
- 7 different custom linear layer implementations
- Standardized initialization across all layers
- Comprehensive training and evaluation utilities
- Comparative analysis framework
"""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Dataset configuration
N_SAMPLES: int = 1_000  # Number of synthetic samples to generate
INPUT_DIM: int = 10     # Input feature dimensionality
N_CLASSES: int = 10     # Number of output classes

# Model configuration
HIDDEN_DIM: int = 256   # Hidden layer size
DROPOUT_RATE: float = 0.20  # Dropout probability for regularization

# Training configuration
LEARNING_RATE: float = 1e-3  # Adam optimizer learning rate
N_EPOCHS: int = 36         # Number of training epochs
BATCH_SIZE: int = 32        # Mini-batch size
TRAIN_SPLIT: float = 0.8    # Training/test split ratio

# Initialization configuration
INIT_RANGE: float = 0.05    # Parameter initialization range [-INIT_RANGE, +INIT_RANGE]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def uniform_init(shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Initialize a tensor with uniform distribution in the range [-INIT_RANGE, +INIT_RANGE].
    
    This controlled initialization helps ensure stable training across different
    layer types by preventing extreme parameter values.
    
    Args:
        shape: Shape of the tensor to initialize
        
    Returns:
        Initialized tensor with uniform distribution
    """
    tensor = torch.empty(*shape)
    return nn.init.uniform_(tensor, -INIT_RANGE, INIT_RANGE)


def create_synthetic_dataset(
    n_samples: int, 
    input_dim: int, 
    n_classes: int, 
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic classification dataset with meaningful structure.
    
    The dataset is created by:
    1. Generating random input features from a standard normal distribution
    2. Computing a weighted combination of the first 5 features as a score
    3. Converting the continuous score into discrete class labels using quantiles
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Number of input features
        n_classes: Number of output classes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels) tensors
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random input features
    X = torch.randn(n_samples, input_dim)
    
    # Create a meaningful relationship between features and target
    # Using a weighted combination of the first 5 features
    score = (
        0.6 * X[:, 0] +   # Strong positive influence
        0.4 * X[:, 1] +   # Moderate positive influence
        -0.3 * X[:, 2] +  # Moderate negative influence
        0.2 * X[:, 3] +   # Weak positive influence
        -0.1 * X[:, 4]    # Weak negative influence
    )
    
    # Convert continuous scores to discrete class labels using quantile-based binning
    # This ensures roughly equal class distribution
    percentile_bins = torch.quantile(score, torch.linspace(0, 1, n_classes + 1))
    y = torch.bucketize(score, boundaries=percentile_bins[1:-1])
    
    return X, y


# ============================================================================
# CUSTOM LINEAR LAYER IMPLEMENTATIONS
# ============================================================================

class StandardLinear(nn.Module):
    """
    Standard linear transformation: output = input @ weight.T + bias
    
    This serves as a baseline for comparison with other custom layers.
    Implements the standard affine transformation used in typical neural networks.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the standard linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weight matrix and bias vector
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the standard linear layer.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return torch.matmul(x, self.weight.T) + self.bias


class QuadraticLinear(nn.Module):
    """
    Quadratic transformation: output = input @ (weight²).T + bias²
    
    This layer applies element-wise squaring to both weights and biases,
    introducing non-linearity through parameter transformation rather than
    activation functions. This can help capture quadratic relationships in data.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the quadratic linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Note: weights and biases will be squared during forward pass
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quadratic parameter transformation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Square the weights and biases to introduce non-linearity
        squared_weight = self.weight ** 2
        squared_bias = self.bias ** 2
        return torch.matmul(x, squared_weight.T) + squared_bias


class TrigonometricLinear(nn.Module):
    """
    Trigonometric transformation: output = sin(input @ weight.T) + cos(bias)
    
    This layer introduces periodic non-linearities through trigonometric functions.
    It can be particularly useful for data with periodic patterns or when you want
    to introduce bounded, oscillatory behavior in the network.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the trigonometric linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with trigonometric transformations.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Apply sine to the linear transformation and cosine to the bias
        linear_output = torch.matmul(x, self.weight.T)
        return torch.sin(linear_output) + torch.cos(self.bias)


class ExponentialLinear(nn.Module):
    """
    Exponential transformation: output = exp(α * (input @ weight.T)) * β + bias
    
    This layer introduces exponential growth/decay patterns. The α parameter
    controls the rate of exponential change, while β acts as a scaling factor.
    Useful for modeling phenomena with exponential relationships.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        alpha: float = 0.1, 
        beta: float = 1.0
    ):
        """
        Initialize the exponential linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            alpha: Exponential scaling factor (controls growth rate)
            beta: Output scaling factor
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha  # Controls exponential growth rate
        self.beta = beta    # Output scaling factor
        
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with exponential transformation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Compute linear transformation
        z = torch.matmul(x, self.weight.T)
        # Apply exponential transformation with scaling
        return torch.exp(self.alpha * z) * self.beta + self.bias


class PolynomialLinear(nn.Module):
    """
    Polynomial transformation: output = a*z³ + b*z² + c*z + bias
    where z = input @ weight.T
    
    This layer implements a cubic polynomial transformation, allowing the network
    to capture complex polynomial relationships in the data. The learnable
    coefficients (a, b, c) allow the model to adapt the polynomial shape.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the polynomial linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
        
        # Polynomial coefficients (learnable parameters)
        self.coeff_cubic = nn.Parameter(uniform_init((output_dim,)))     # a (cubic term)
        self.coeff_quadratic = nn.Parameter(uniform_init((output_dim,))) # b (quadratic term)
        self.coeff_linear = nn.Parameter(uniform_init((output_dim,)))    # c (linear term)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with polynomial transformation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Compute base linear transformation
        z = torch.matmul(x, self.weight.T)
        
        # Apply cubic polynomial: a*z³ + b*z² + c*z + bias
        return (self.coeff_cubic * z**3 + 
                self.coeff_quadratic * z**2 + 
                self.coeff_linear * z + 
                self.bias)


class GaussianLinear(nn.Module):
    """
    Gaussian RBF-like transformation: output_j = exp(-||x - w_j||² / σ²) + bias_j
    
    This layer implements Radial Basis Function (RBF) behavior using Gaussian kernels.
    Each output neuron acts as a Gaussian centered at the corresponding weight vector.
    Useful for modeling local, smooth non-linear relationships.
    """
    
    def __init__(self, input_dim: int, output_dim: int, sigma: float = 1.0):
        """
        Initialize the Gaussian RBF linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            sigma: Standard deviation of Gaussian kernels (controls width)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma  # Controls the width of Gaussian kernels
        
        # Weight vectors serve as Gaussian centers
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Gaussian RBF transformation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Compute squared Euclidean distances between input and weight centers
        # x: [batch_size, input_dim] -> [batch_size, 1, input_dim]
        # weight: [output_dim, input_dim] -> [1, output_dim, input_dim]
        x_expanded = x.unsqueeze(1)          # [batch_size, 1, input_dim]
        weight_expanded = self.weight.unsqueeze(0)  # [1, output_dim, input_dim]
        
        # Compute difference vectors
        diff = x_expanded - weight_expanded   # [batch_size, output_dim, input_dim]
        
        # Compute squared distances
        squared_distances = torch.sum(diff**2, dim=2)  # [batch_size, output_dim]
        
        # Apply Gaussian kernel and add bias
        return torch.exp(-squared_distances / (self.sigma ** 2)) + self.bias


class FourierLinear(nn.Module):
    """
    Fourier-inspired transformation: output = A*cos(ω*(input @ weight.T) + φ) + bias
    
    This layer implements a Fourier-like transformation with learnable amplitude (A),
    frequency (ω), and phase (φ) parameters. It can capture periodic patterns and
    oscillatory behavior in the data.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the Fourier-inspired linear layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Standard transformation parameters
        self.weight = nn.Parameter(uniform_init((output_dim, input_dim)))
        self.bias = nn.Parameter(uniform_init((output_dim,)))
        
        # Fourier transformation parameters
        self.amplitude = nn.Parameter(uniform_init((output_dim,)))  # A (amplitude)
        self.frequency = nn.Parameter(uniform_init((output_dim,)))  # ω (frequency)
        self.phase = nn.Parameter(uniform_init((output_dim,)))      # φ (phase)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Fourier transformation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Compute base linear transformation
        z = torch.matmul(x, self.weight.T)
        
        # Apply Fourier transformation: A*cos(ω*z + φ) + bias
        fourier_output = self.amplitude * torch.cos(self.frequency * z + self.phase)
        return fourier_output + self.bias


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class CustomLayerNetwork(nn.Module):
    """
    A three-layer neural network that incorporates custom linear transformations.
    
    Architecture:
    Input -> Standard Linear -> ReLU -> Dropout -> Custom Layer -> Dropout -> Output Linear
    
    This architecture allows for direct comparison of different custom layer types
    while maintaining consistent network structure and capacity.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        layer_type: str,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the custom layer network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
            layer_type: Type of custom layer to use
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_type = layer_type
        
        # First layer: standard linear transformation with ReLU activation
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Custom layer selection
        custom_layer_map = {
            'standard': StandardLinear,
            'quadratic': QuadraticLinear,
            'trigonometric': TrigonometricLinear,
            'exponential': ExponentialLinear,
            'polynomial': PolynomialLinear,
            'gaussian': GaussianLinear,
            'fourier': FourierLinear
        }
        
        if layer_type not in custom_layer_map:
            raise ValueError(f"Unknown layer type: {layer_type}. "
                           f"Available types: {list(custom_layer_map.keys())}")
        
        # Custom transformation layer
        self.custom_layer = custom_layer_map[layer_type](hidden_dim, hidden_dim)
        
        # Output layer: standard linear transformation for classification
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output logits of shape [batch_size, output_dim]
        """
        # Input layer with ReLU activation
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # Custom transformation layer
        x = self.custom_layer(x)
        x = self.dropout(x)
        
        # Output layer (no activation - raw logits for CrossEntropyLoss)
        return self.output_layer(x)


# ============================================================================
# TRAINING AND EVALUATION UTILITIES
# ============================================================================

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    num_epochs: int = 10,
    verbose: bool = True
) -> None:
    """
    Train a model for the specified number of epochs.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        verbose: Whether to print training progress
    """
    model.train()  # Set model to training mode
    
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for reporting
            total_loss += loss.item()
            num_batches += 1
        
        # Report epoch progress
        if verbose:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch:3d}/{num_epochs}: Loss = {avg_loss:.4f}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_batches = 0
    
    for batch_x, batch_y in test_loader:
        # Forward pass (no gradient computation)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate accuracy
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == batch_y).sum().item()
        total_samples += batch_y.size(0)
    
    # Calculate final metrics
    average_loss = total_loss / num_batches
    accuracy = 100.0 * correct_predictions / total_samples
    
    return average_loss, accuracy


def create_data_loaders(
    features: torch.Tensor, 
    labels: torch.Tensor, 
    train_split: float = 0.8, 
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing data loaders from feature and label tensors.
    
    Args:
        features: Input feature tensor
        labels: Target label tensor
        train_split: Fraction of data to use for training
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Calculate split index
    num_samples = features.size(0)
    split_idx = int(train_split * num_samples)
    
    # Split data
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    test_features = features[split_idx:]
    test_labels = labels[split_idx:]
    
    # Create datasets
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


# ============================================================================
# EXPERIMENTAL FRAMEWORK
# ============================================================================

def run_layer_comparison_experiment() -> Dict[str, Dict[str, float]]:
    """
    Run a comprehensive comparison of all custom layer types.
    
    This function:
    1. Generates synthetic data
    2. Trains models with each custom layer type
    3. Evaluates performance on test data
    4. Returns comparative results
    
    Returns:
        Dictionary mapping layer types to their performance metrics
    """
    print("=" * 80)
    print("CUSTOM LINEAR LAYERS COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Dataset: {N_SAMPLES} samples, {INPUT_DIM} features, {N_CLASSES} classes")
    print(f"Architecture: {INPUT_DIM} -> {HIDDEN_DIM} -> {HIDDEN_DIM} -> {N_CLASSES}")
    print(f"Training: {N_EPOCHS} epochs, learning rate {LEARNING_RATE}")
    print("=" * 80)
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    features, labels = create_synthetic_dataset(N_SAMPLES, INPUT_DIM, N_CLASSES)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        features, labels, TRAIN_SPLIT, BATCH_SIZE
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Define layer types to test
    layer_types = [
        'standard', 'quadratic', 'trigonometric', 'exponential',
        'polynomial', 'gaussian', 'fourier'
    ]
    
    # Store results for comparison
    results = {}
    
    # Train and evaluate each layer type
    for layer_type in layer_types:
        print(f"\n{'-' * 60}")
        print(f"TRAINING {layer_type.upper()} LAYER")
        print(f"{'-' * 60}")
        
        # Create model with current layer type
        model = CustomLayerNetwork(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=N_CLASSES,
            layer_type=layer_type,
            dropout_rate=DROPOUT_RATE
        )
        
        # Setup training components
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        train_model(model, train_loader, criterion, optimizer, N_EPOCHS)
        
        # Evaluate the model
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        
        # Store results
        results[layer_type] = {
            'loss': test_loss,
            'accuracy': test_accuracy
        }
        
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
    
    return results


def display_results_summary(results: Dict[str, Dict[str, float]]) -> None:
    """
    Display a formatted summary of experimental results.
    
    Args:
        results: Dictionary mapping layer types to performance metrics
    """
    print(f"\n{'=' * 80}")
    print("EXPERIMENTAL RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    # Create formatted table
    print(f"{'Layer Type':<15} {'Test Loss':<12} {'Accuracy (%)':<15} {'Rank':<8}")
    print(f"{'-' * 80}")
    
    # Sort by accuracy for ranking
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['accuracy'], 
        reverse=True
    )
    
    for rank, (layer_type, metrics) in enumerate(sorted_results, 1):
        print(f"{layer_type.capitalize():<15} "
              f"{metrics['loss']:<12.4f} "
              f"{metrics['accuracy']:<15.2f} "
              f"#{rank}")
    
    print(f"{'=' * 80}")
    
    # Highlight best performers
    best_accuracy_layer = sorted_results[0]
    best_loss_layer = min(results.items(), key=lambda x: x[1]['loss'])
    
    print(f"\n🏆 BEST ACCURACY: {best_accuracy_layer[0].capitalize()} "
          f"({best_accuracy_layer[1]['accuracy']:.2f}%)")
    print(f"📉 LOWEST LOSS: {best_loss_layer[0].capitalize()} "
          f"({best_loss_layer[1]['loss']:.4f})")
    
    # Calculate performance statistics
    accuracies = [result['accuracy'] for result in results.values()]
    losses = [result['loss'] for result in results.values()]
    
    print(f"\n📊 PERFORMANCE STATISTICS:")
    print(f"   Accuracy Range: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
    print(f"   Accuracy Std Dev: {np.std(accuracies):.2f}%")
    print(f"   Loss Range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"   Loss Std Dev: {np.std(losses):.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete experimental pipeline.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the comparison experiment
    results = run_layer_comparison_experiment()
    
    # Display results
    display_results_summary(results)
    
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
