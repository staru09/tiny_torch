import numpy as np

from src.tensor import Tensor

EPSILON = 1e-7


class Function:
    """Base class for backward (gradient) functions."""

    def __init__(self, *tensors):
        self.saved_tensors = tensors

    def apply(self, grad_output):
        raise NotImplementedError("Each Function must implement apply()")

def _reduce_broadcast_grad(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i in range(len(original_shape)):
        if original_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class AddBackward(Function):
    """Gradient computation for tensor addition."""

    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)
        return (grad_a, grad_b)

class MulBackward(Function):
    """Gradient computation for tensor multiplication."""

    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)
        return (grad_a, grad_b)

class SubBackward(Function):
    """Gradient computation for tensor subtraction."""

    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)
        return (grad_a, grad_b)

class DivBackward(Function):
    """Gradient computation for tensor division."""

    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output / b.data
            else:
                grad_a = grad_output / b
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output * a.data / b.data ** 2
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)
        return (grad_a, grad_b)

class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None
        if isinstance(a, Tensor) and a.requires_grad:
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data, -2, -1)
            else:
                b_T = b.data.T
            grad_a = np.matmul(grad_output, b_T)
        if isinstance(b, Tensor) and b.requires_grad:
            if a.data.ndim >= 2:
                a_T = np.swapaxes(a.data, -2, -1)
            else:
                a_T = a.data.T
            grad_b = np.matmul(a_T, grad_output)
        return (grad_a, grad_b)

class TransposeBackward(Function):
    """Gradient computation for transpose operation."""

    def __init__(self, tensor, dim0, dim1):
        """
        Args:
            tensor: Input tensor
            dim0: First dimension to swap (None for default)
            dim1: Second dimension to swap (None for default)
        """
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None
        if isinstance(x, Tensor) and x.requires_grad:
            if self.dim0 is None and self.dim1 is None:
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2], axes[-1] = (axes[-1], axes[-2])
                    grad_x = np.transpose(grad_output, axes)
            else:
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = (axes[self.dim1], axes[self.dim0])
                grad_x = np.transpose(grad_output, axes)
        return (grad_x,)

class PermuteBackward(Function):
    """Gradient computation for arbitrary axis permutation (general transpose)."""

    def __init__(self, tensor, axes):
        """
        Args:
            tensor: Input tensor
            axes: Tuple of axis indices defining the permutation
        """
        super().__init__(tensor)
        self.axes = axes
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None
        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = np.transpose(grad_output, self.inverse_axes)
        return (grad_x,)

class SliceBackward(Function):
    """Gradient computation for tensor slicing/indexing operations."""

    def __init__(self, tensor, key):
        """
        Args:
            tensor: Original tensor being sliced
            key: Slicing key (index, slice, tuple of slices, etc.)
        """
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        grad_input = None
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            grad_input = np.zeros(self.original_shape, dtype=np.float32)
            grad_input[self.key] = grad_output
        return (grad_input,)

class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    def __init__(self, tensor, original_shape):
        """
        Args:
            tensor: Input tensor
            original_shape: Shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None
        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = grad_output.reshape(self.original_shape)
        return (grad_x,)

class SumBackward(Function):
    """Gradient computation for tensor sum."""

    def __init__(self, tensor, axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            if self.axis is not None and (not self.keepdims):
                grad_output = np.expand_dims(grad_output, axis=self.axis)
            return (np.ones_like(tensor.data) * grad_output,)
        return (None,)


class ReLUBackward(Function):
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            relu_grad = (tensor.data > 0).astype(np.float32)
            return grad_output * relu_grad,
        return None,


class SigmoidBackward(Function):
    def __init__(self, input_tensor, output_tensor):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return grad_output * sigmoid_grad,
        return None,


class TanhBackward(Function):
    def __init__(self, input_tensor, output_tensor):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            tanh_grad = 1 - self.output_data ** 2
            return grad_output * tanh_grad,
        return None,


class SoftmaxBackward(Function):
    def __init__(self, input_tensor, output_tensor, dim=-1):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            sum_term = np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)
            grad_x = self.output_data * (grad_output - sum_term)
            return (grad_x,)
        return (None,)


class GELUBackward(Function):
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            sig = 1.0 / (1.0 + np.exp(-1.702 * x))
            gelu_grad = sig + x * 1.702 * sig * (1.0 - sig)
            return (grad_output * gelu_grad,)
        return (None,)


class MSEBackward(Function):
    def __init__(self, predictions, targets):
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        predictions, = self.saved_tensors
        if isinstance(predictions, Tensor) and predictions.requires_grad:
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples
            return grad * grad_output,
        return None,


class BCEBackward(Function):
    def __init__(self, predictions, targets):
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        predictions, = self.saved_tensors
        if isinstance(predictions, Tensor) and predictions.requires_grad:
            p = np.clip(predictions.data, EPSILON, 1 - EPSILON)
            y = self.targets_data
            grad = (p - y) / (p * (1 - p) * self.num_samples)
            return grad * grad_output,
        return None,


def _stable_softmax(logits_data):
    max_logits = np.max(logits_data, axis=1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _one_hot_encode(targets, batch_size, num_classes):
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets] = 1.0
    return one_hot


class CrossEntropyBackward(Function):
    def __init__(self, logits, targets):
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        logits, = self.saved_tensors
        if isinstance(logits, Tensor) and logits.requires_grad:
            softmax = _stable_softmax(logits.data)
            one_hot = _one_hot_encode(self.targets_data, self.batch_size, self.num_classes)
            grad = (softmax - one_hot) / self.batch_size
            return grad * grad_output,
        return None,


_GRAD_TRACKING_ENABLED = True


def is_grad_enabled():
    return _GRAD_TRACKING_ENABLED


class no_grad:
    """Context manager that disables gradient tracking."""

    def __enter__(self):
        global _GRAD_TRACKING_ENABLED
        self._prev_state = _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = self._prev_state
        return False

def enable_autograd(quiet=False):
    if hasattr(Tensor, '_autograd_enabled'):
        return
    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None
    Tensor.__init__ = gradient_aware_init
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__
    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    def _get_requires_grad(tensor):
        if not _GRAD_TRACKING_ENABLED:
            return False
        return getattr(tensor, 'requires_grad', False) if isinstance(tensor, Tensor) else False

    def _ensure_grad_attrs(tensor):
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, 'requires_grad'):
                tensor.requires_grad = False
            if not hasattr(tensor, 'grad'):
                tensor.grad = None

    def tracked_add(self, other):
        _ensure_grad_attrs(self)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)
        result = _original_add(self, other)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)
        return result

    def tracked_mul(self, other):
        _ensure_grad_attrs(self)
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)
        result = _original_mul(self, other)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)
        return result

    def tracked_matmul(self, other):
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)
        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)
        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        _ensure_grad_attrs(self)
        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)
        return result

    def tracked_reshape(self, *shape):
        _ensure_grad_attrs(self)
        original_shape = self.shape
        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)
        return result

    def tracked_sub(self, other):
        _ensure_grad_attrs(self)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)
        result = _original_sub(self, other)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other)
        return result

    def tracked_div(self, other):
        _ensure_grad_attrs(self)
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)
        result = _original_div(self, other)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other)
        return result

    def tracked_getitem(self, key):
        _ensure_grad_attrs(self)
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)
        return result

    def sum_op(self, axis=None, keepdims=False):
        _ensure_grad_attrs(self)
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SumBackward(self, axis=axis, keepdims=keepdims)
        return result

    def backward(self, gradient=None, retain_graph=False):
        _ensure_grad_attrs(self)
        if not _get_requires_grad(self):
            return
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(f'backward() called on non-scalar tensor without gradient argument.\n  Tensor shape: {self.shape}\n  Issue: For non-scalar outputs, you must provide the gradient from the next layer.\n  Fix: Call backward(gradient) with the gradient tensor from the loss function.')
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        if gradient.shape != self.grad.shape:
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)
        self.grad += gradient
        grad_fn = getattr(self, '_grad_fn', None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and (grad is not None):
                    tensor.backward(grad, retain_graph=retain_graph)
            if not retain_graph:
                self._grad_fn = None

    def zero_grad(self):
        self.grad = None
    Tensor.__add__ = tracked_add
    Tensor.__sub__ = tracked_sub
    Tensor.__mul__ = tracked_mul
    Tensor.__truediv__ = tracked_div
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad
    try:
        from src.activations import Sigmoid, ReLU, Softmax, GELU, Tanh
        from src.loss import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_tanh_forward = Tanh.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self, x):
            result_data = 1.0 / (1.0 + np.exp(-x.data))
            result = Tensor(result_data)
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)
            return result

        def tracked_tanh_forward(self, x):
            result_data = np.tanh(x.data)
            result = Tensor(result_data)
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = TanhBackward(x, result)
            return result

        def tracked_relu_forward(self, x):
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)
            return result

        def tracked_softmax_forward(self, x, dim=-1):
            result = _original_softmax_forward(self, x, dim=dim)
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SoftmaxBackward(x, result, dim)
            return result

        def tracked_gelu_forward(self, x):
            result = _original_gelu_forward(self, x)
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = GELUBackward(x)
            return result

        def tracked_bce_forward(self, predictions, targets):
            eps = EPSILON
            clamped_preds = np.clip(predictions.data, eps, 1 - eps)
            log_preds = np.log(clamped_preds)
            log_one_minus_preds = np.log(1 - clamped_preds)
            bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)
            bce_loss = np.mean(bce_per_sample)
            result = Tensor(bce_loss)
            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)
            return result

        def tracked_mse_forward(self, predictions, targets):
            diff = predictions.data - targets.data
            squared_diff = diff ** 2
            mse = np.mean(squared_diff)
            result = Tensor(mse)
            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)
            return result

        def tracked_ce_forward(self, logits, targets):
            from src.loss import log_softmax
            log_probs = log_softmax(logits, dim=-1)
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
            ce_loss = -np.mean(selected_log_probs)
            result = Tensor(ce_loss)
            if _GRAD_TRACKING_ENABLED and logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)
            return result
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        Tanh.forward = tracked_tanh_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward
    except ImportError:
        pass
    Tensor._autograd_enabled = True
    if not quiet:
        print("Autograd enabled on Tensor.")

enable_autograd(quiet=True)
