from src.tensor import Tensor
import numpy as np

EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log-softmax with numerical stability.

    TODO: Implement numerically stable log-softmax using the log-sum-exp trick

    APPROACH:
    1. Find maximum along dimension (for stability)
    2. Subtract max from input (prevents overflow)
    3. Compute log(sum(exp(shifted_input)))
    4. Return input - max - log_sum_exp

    EXAMPLE:
    >>> logits = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    >>> result = log_softmax(logits, dim=-1)
    >>> print(result.shape)
    (2, 3)

    HINT: Use np.max(x.data, axis=dim, keepdims=True) to preserve dimensions
"""

    max_val = np.max(x.data, axis=dim,keepdims=True)

    shift = x.data - max_val

    log_sum_exp = np.log(np.sum(np.exp(shift), axis=dim, keepdims=True))

    result = x.data - max_val - log_sum_exp

    return Tensor(result)
class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""

    def __init__(self):
        """Initialize cross-entropy loss function."""
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between logits and target class indices.

        TODO: Implement cross-entropy loss with numerical stability

        APPROACH:
        1. Compute log-softmax of logits (numerically stable)
        2. Select log-probabilities for correct classes
        3. Return negative mean of selected log-probabilities

        EXAMPLE:
        >>> loss_fn = CrossEntropyLoss()
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.8]])  # 2 samples, 3 classes
        >>> targets = Tensor([0, 1])  # First sample is class 0, second is class 1
        >>> loss = loss_fn(logits, targets)
        >>> print(f"Cross-Entropy Loss: {loss.data:.4f}")

        """

        log_probs = log_softmax(logits,dim=-1)

        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)

    def __call__(self,logits,targets):
        return self.forward(logits,targets)

    def backward(self):
        pass

class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        """Initialize MSE loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute mean squared error between predictions and targets.

        TODO: Implement MSE loss calculation

        APPROACH:
        1. Compute difference: predictions - targets
        2. Square the differences: diff²
        3. Take mean across all elements
        """

        diff = predictions.data - targets.data

        squared_diff = diff**2

        mse = np.mean(squared_diff)

        return Tensor(mse)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function."""
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        pass

class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function."""
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.

        TODO: Implement binary cross-entropy with numerical stability

        APPROACH:
        1. Clamp predictions to avoid log(0) and log(1)
        2. Compute: -(targets * log(predictions) + (1-targets) * log(1-predictions))
        3. Return mean across all samples

        EXAMPLE:
        >>> loss_fn = BinaryCrossEntropyLoss()
        >>> predictions = Tensor([0.9, 0.1, 0.7, 0.3])  # Probabilities between 0 and 1
        >>> targets = Tensor([1.0, 0.0, 1.0, 0.0])      # Binary labels
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Binary Cross-Entropy Loss: {loss.data:.4f}")
        """

        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        log_preds = np.log(clamped_preds)

        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)

    def __call__(self,predictions,target):

        return self.forward(predictions,target)

    def backward(self):
        pass 