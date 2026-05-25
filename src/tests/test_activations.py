import warnings
import numpy as np
from src.activations import GELU, ReLU, Sigmoid, Softmax, Tanh
from src.tensor import Tensor
TOLERANCE = 1e-05

def test_unit_sigmoid():
    """Test Sigmoid implementation."""
    sigmoid = Sigmoid()
    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert np.allclose(result.data, [0.5]), f'sigmoid(0) should be 0.5, got {result.data}'
    x = Tensor([-10, -1, 0, 1, 10])
    result = sigmoid.forward(x)
    assert np.all(result.data > 0) and np.all(result.data < 1), 'All sigmoid outputs should be in (0, 1)'
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    x = Tensor([-1000, 1000])
    result = sigmoid.forward(x)
    assert np.allclose(result.data[0], 0, atol=TOLERANCE)
    assert np.allclose(result.data[1], 1, atol=TOLERANCE)
    warnings.filterwarnings('default', category=RuntimeWarning)

def test_unit_relu():
    """Test ReLU implementation."""
    relu = ReLU()
    x = Tensor([-2, -1, 0, 1, 2])
    result = relu.forward(x)
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(result.data, expected), f'ReLU failed, expected {expected}, got {result.data}'
    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert np.allclose(result.data, [0, 0, 0]), 'ReLU should zero all negative values'
    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert np.allclose(result.data, [1, 3, 5]), 'ReLU should preserve all positive values'
    x = Tensor([-1, -2, -3, 1])
    result = relu.forward(x)
    zeros = np.sum(result.data == 0)
    assert zeros == 3, f'ReLU should create sparsity, got {zeros} zeros out of 4'

def test_unit_tanh():
    """Test Tanh implementation."""
    tanh = Tanh()
    x = Tensor([0.0])
    result = tanh.forward(x)
    assert np.allclose(result.data, [0.0]), f'tanh(0) should be 0, got {result.data}'
    x = Tensor([-10, -1, 0, 1, 10])
    result = tanh.forward(x)
    assert np.all(result.data >= -1) and np.all(result.data <= 1), 'All tanh outputs should be in [-1, 1]'
    x = Tensor([2.0])
    pos_result = tanh.forward(x)
    x_neg = Tensor([-2.0])
    neg_result = tanh.forward(x_neg)
    assert np.allclose(pos_result.data, -neg_result.data), 'tanh should be symmetric: tanh(-x) = -tanh(x)'
    x = Tensor([-1000, 1000])
    result = tanh.forward(x)
    assert np.allclose(result.data[0], -1, atol=TOLERANCE)
    assert np.allclose(result.data[1], 1, atol=TOLERANCE)

def test_unit_gelu():
    """Test GELU implementation."""
    gelu = GELU()
    x = Tensor([0.0])
    result = gelu.forward(x)
    assert np.allclose(result.data, [0.0], atol=TOLERANCE)
    x = Tensor([1.0])
    result = gelu.forward(x)
    assert result.data[0] > 0.8
    x = Tensor([-1.0])
    result = gelu.forward(x)
    assert result.data[0] < 0 and result.data[0] > -0.2
    x = Tensor([-0.001, 0.0, 0.001])
    result = gelu.forward(x)
    diff1 = abs(result.data[1] - result.data[0])
    diff2 = abs(result.data[2] - result.data[1])
    assert diff1 < 0.01 and diff2 < 0.01, 'GELU should be smooth around zero'

def test_unit_softmax():
    """Test Softmax implementation."""
    softmax = Softmax()
    x = Tensor([1, 2, 3])
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), f'Softmax should sum to 1, got {np.sum(result.data)}'
    assert np.all(result.data > 0), 'All softmax values should be positive'
    assert np.all(result.data < 1), 'All softmax values should be less than 1'
    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(result.data)
    assert max_input_idx == max_output_idx, 'Largest input should get largest softmax output'
    x = Tensor([1000, 1001, 1002])
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), 'Softmax should handle large numbers'
    assert not np.any(np.isnan(result.data)), 'Softmax should not produce NaN'
    assert not np.any(np.isinf(result.data)), 'Softmax should not produce infinity'
    x = Tensor([[1, 2], [3, 4]])
    result = softmax.forward(x, dim=-1)
    assert result.shape == (2, 2), 'Softmax should preserve input shape'
    row_sums = np.sum(result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), 'Each row should sum to 1'

def test_module():
    test_unit_sigmoid()
    test_unit_relu()
    test_unit_tanh()
    test_unit_gelu()
    test_unit_softmax()
    test_data = Tensor([[1, -1], [2, -2]])
    activations = [Sigmoid(), ReLU(), Tanh(), GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        assert result.shape == test_data.shape, f'Shape not preserved by {activation.__class__.__name__}'
        assert isinstance(result, Tensor), f'Output not Tensor from {activation.__class__.__name__}'
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    softmax = Softmax()
    result_last = softmax(data_3d, dim=-1)
    assert result_last.shape == (2, 2, 3), 'Softmax should preserve shape'
    last_dim_sums = np.sum(result_last.data, axis=-1)
    assert np.allclose(last_dim_sums, 1.0), 'Last dimension should sum to 1'
    x = Tensor([[-1, 0, 1, 2]])
    relu = ReLU()
    hidden = relu.forward(x)
    softmax = Softmax()
    output = softmax.forward(hidden)
    assert hidden.data[0, 0] == 0, 'ReLU should zero negative input'
    assert np.allclose(np.sum(output.data), 1.0), 'Final output should be probability distribution'
