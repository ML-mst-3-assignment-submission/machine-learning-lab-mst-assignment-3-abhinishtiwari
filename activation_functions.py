"""
activation_functions.py
------------------------
Implement and visualise common neural network activation functions:
    Sigmoid, ReLU, Tanh, Softmax, Leaky ReLU, ELU, Swish
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# FUNCTION DEFINITIONS
# ─────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Maps any real number to (0, 1). Used in output layer for binary classification."""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit. Most popular hidden-layer activation."""
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent. Output in (-1, 1). Better than sigmoid for hidden layers."""
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Converts a vector into a probability distribution (multi-class output)."""
    exp_x = np.exp(x - np.max(x))   # numerical stability trick
    return exp_x / exp_x.sum(axis=0)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU – avoids the 'dying ReLU' problem by allowing small negative values."""
    return np.where(x > 0, x, alpha * x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit – smooth version of ReLU."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def swish(x: np.ndarray) -> np.ndarray:
    """Swish (x * sigmoid(x)) – used in Google's EfficientNet. Often beats ReLU."""
    return x * sigmoid(x)


# ─────────────────────────────────────────────
# DERIVATIVES
# ─────────────────────────────────────────────

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_activations(save_path: str = "outputs/activation_functions.png") -> None:
    """Plot all activation functions side by side and save the figure."""
    x = np.linspace(-10, 10, 500)

    funcs = {
        "Sigmoid":    sigmoid(x),
        "ReLU":       relu(x),
        "Tanh":       tanh(x),
        "Leaky ReLU": leaky_relu(x),
        "ELU":        elu(x),
        "Swish":      swish(x),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    colors = ["royalblue", "tomato", "seagreen", "orange", "purple", "darkred"]

    for ax, (name, y), color in zip(axes, funcs.items(), colors):
        ax.plot(x, y, color=color, linewidth=2)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 3)

    plt.suptitle("Neural Network Activation Functions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {save_path}")


def plot_derivatives(save_path: str = "outputs/activation_derivatives.png") -> None:
    """Plot activation functions alongside their derivatives."""
    x = np.linspace(-6, 6, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, fn, dfn, color) in zip(axes, [
        ("Sigmoid", sigmoid, sigmoid_derivative, "royalblue"),
        ("ReLU",    relu,    relu_derivative,    "tomato"),
        ("Tanh",    tanh,    tanh_derivative,    "seagreen"),
    ]):
        ax.plot(x, fn(x),  color=color, linewidth=2, label="f(x)")
        ax.plot(x, dfn(x), color=color, linewidth=2, linestyle="--", label="f'(x)")
        ax.set_title(f"{name} & Derivative")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"Derivatives plot saved to {save_path}")


def demo_values() -> None:
    """Print sample activation outputs for a small array of values."""
    values = np.array([-3, -1, 0, 1, 3], dtype=float)
    print(f"\nInput values : {values}")
    print(f"Sigmoid      : {sigmoid(values).round(4)}")
    print(f"ReLU         : {relu(values)}")
    print(f"Tanh         : {tanh(values).round(4)}")
    print(f"Leaky ReLU   : {leaky_relu(values)}")
    print(f"ELU          : {elu(values).round(4)}")
    print(f"Swish        : {swish(values).round(4)}")

    sample_vec = np.array([2.0, 1.0, 0.1])
    print(f"\nSoftmax({sample_vec}) = {softmax(sample_vec).round(4)}")
    print(f"  Sum = {softmax(sample_vec).sum():.4f} (should be 1.0)")


if __name__ == "__main__":
    demo_values()
    plot_activations()
    plot_derivatives()
