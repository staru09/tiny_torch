import numpy as np

import src.autograd  # noqa: F401 - enables gradients on import

from src.autograd import (
    AddBackward,
    CrossEntropyBackward,
    DivBackward,
    MatmulBackward,
    MulBackward,
    SubBackward,
    _one_hot_encode,
    _reduce_broadcast_grad,
    _stable_softmax,
)
from src.tensor import Tensor
rng = np.random.default_rng(7)

def test_unit_reduce_broadcast_grad():
    """Test _reduce_broadcast_grad helper function."""
    grad = np.ones((32, 128))
    original_shape = (128,)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (128,), f'Expected (128,), got {reduced.shape}'
    assert np.allclose(reduced, np.ones(128) * 32), 'Should sum over batch dimension'
    grad = np.ones((10, 5))
    original_shape = (10, 1)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 1), f'Expected (10, 1), got {reduced.shape}'
    assert np.allclose(reduced, np.ones((10, 1)) * 5), 'Should sum over singleton axis'
    grad = np.ones((32, 10, 5))
    original_shape = (10, 1)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 1), f'Expected (10, 1), got {reduced.shape}'
    expected = np.ones((10, 1)) * 32 * 5
    assert np.allclose(reduced, expected), 'Should handle multiple reductions'
    grad = np.ones((10, 5))
    original_shape = (10, 5)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 5), f'Expected (10, 5), got {reduced.shape}'
    assert np.allclose(reduced, grad), 'Should return unchanged when shapes match'
    grad = np.ones((32, 128))
    original_shape = ()
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (), f'Expected scalar, got {reduced.shape}'
    assert np.allclose(reduced, 32 * 128), 'Should sum to scalar'

def test_unit_function_classes():
    """Test Function classes."""
    a = Tensor([1, 2, 3])
    a.requires_grad = True
    b = Tensor([4, 5, 6])
    b.requires_grad = True
    add_func = AddBackward(a, b)
    grad_output = np.array([1, 1, 1])
    grad_a, grad_b = add_func.apply(grad_output)
    assert np.allclose(grad_a, grad_output), f'AddBackward grad_a failed: {grad_a}'
    assert np.allclose(grad_b, grad_output), f'AddBackward grad_b failed: {grad_b}'
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(grad_output)
    assert np.allclose(grad_a, b.data), f'MulBackward grad_a failed: {grad_a}'
    assert np.allclose(grad_b, a.data), f'MulBackward grad_b failed: {grad_b}'
    a_mat = Tensor([[1, 2], [3, 4]])
    a_mat.requires_grad = True
    b_mat = Tensor([[5, 6], [7, 8]])
    b_mat.requires_grad = True
    matmul_func = MatmulBackward(a_mat, b_mat)
    grad_output = np.ones((2, 2))
    grad_a, grad_b = matmul_func.apply(grad_output)
    assert grad_a.shape == a_mat.shape, f'MatmulBackward grad_a shape: {grad_a.shape}'
    assert grad_b.shape == b_mat.shape, f'MatmulBackward grad_b shape: {grad_b.shape}'

def test_unit_broadcast_gradients():
    """Test gradient broadcasting reduction."""
    x = Tensor(rng.standard_normal((4, 3)), requires_grad=True)
    bias = Tensor(np.ones(3), requires_grad=True)
    add_func = AddBackward(x, bias)
    grad_output = np.ones((4, 3))
    grad_x, grad_bias = add_func.apply(grad_output)
    assert grad_x.shape == x.data.shape, f'Expected grad_x shape {x.data.shape}, got {grad_x.shape}'
    assert grad_bias.shape == bias.data.shape, f'Expected grad_bias shape {bias.data.shape}, got {grad_bias.shape}'
    expected_bias_grad = np.ones(3) * 4
    assert np.allclose(grad_bias, expected_bias_grad), f'Expected bias grad {expected_bias_grad}, got {grad_bias}'
    x = Tensor(rng.standard_normal((3, 4)), requires_grad=True)
    scalar_val = 5.0
    add_func = AddBackward(x, scalar_val)
    grad_output = np.ones((3, 4))
    grad_x, grad_scalar = add_func.apply(grad_output)
    assert grad_x.shape == x.data.shape, f'Expected grad_x shape {x.data.shape}, got {grad_x.shape}'
    x = Tensor(rng.standard_normal((32, 10, 5)), requires_grad=True)
    y = Tensor(rng.standard_normal((10, 1)), requires_grad=True)
    mul_func = MulBackward(x, y)
    grad_output = np.ones((32, 10, 5))
    grad_x, grad_y = mul_func.apply(grad_output)
    assert grad_x.shape == x.data.shape, f'Expected grad_x shape {x.data.shape}, got {grad_x.shape}'
    assert grad_y.shape == y.data.shape, f'Expected grad_y shape {y.data.shape}, got {grad_y.shape}'
    a = Tensor(rng.standard_normal((8, 16)), requires_grad=True)
    b = Tensor(rng.standard_normal(16), requires_grad=True)
    add_func = AddBackward(a, b)
    grad_a, grad_b = add_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f'AddBackward: Expected (16,), got {grad_b.shape}'
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f'MulBackward: Expected (16,), got {grad_b.shape}'
    sub_func = SubBackward(a, b)
    grad_a, grad_b = sub_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f'SubBackward: Expected (16,), got {grad_b.shape}'
    div_func = DivBackward(a, b)
    grad_a, grad_b = div_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f'DivBackward: Expected (16,), got {grad_b.shape}'
    batch_size, out_features = (32, 128)
    output_grad = rng.standard_normal((batch_size, out_features))
    bias = Tensor(np.zeros(out_features), requires_grad=True)
    add_func = AddBackward(Tensor(np.zeros((batch_size, out_features)), requires_grad=True), bias)
    _, grad_bias = add_func.apply(output_grad)
    assert grad_bias.shape == (out_features,), f'Linear layer bias: Expected ({out_features},), got {grad_bias.shape}'
    expected = output_grad.sum(axis=0)
    assert np.allclose(grad_bias, expected), 'Bias gradient should equal sum over batch dimension'

def test_unit_stable_softmax():
    """Test stable softmax helper."""
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = _stable_softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0), f'Softmax should sum to 1, got {probs.sum(axis=1)}'
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    assert np.allclose(probs, expected), f'Softmax values wrong: {probs}'
    large_logits = np.array([[1000.0, 1001.0, 1002.0]])
    probs_large = _stable_softmax(large_logits)
    assert not np.any(np.isnan(probs_large)), 'Stable softmax should handle large values'
    assert not np.any(np.isinf(probs_large)), 'Stable softmax should not overflow'
    assert np.allclose(probs_large.sum(axis=1), 1.0), 'Large softmax should sum to 1'
    batch_logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch_probs = _stable_softmax(batch_logits)
    assert batch_probs.shape == (3, 2), f'Expected (3, 2), got {batch_probs.shape}'
    assert np.allclose(batch_probs.sum(axis=1), np.ones(3)), 'Each row should sum to 1'

def test_unit_one_hot_encode():
    """Test one-hot encoding helper."""
    targets = np.array([0, 2, 1])
    result = _one_hot_encode(targets, batch_size=3, num_classes=3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    assert np.allclose(result, expected), f'One-hot encoding wrong: {result}'
    result_single = _one_hot_encode(np.array([1]), batch_size=1, num_classes=4)
    assert result_single.shape == (1, 4), f'Expected (1, 4), got {result_single.shape}'
    assert result_single[0, 1] == 1.0, 'Target class should be 1.0'
    assert result_single.sum() == 1.0, 'Should have exactly one 1.0'
    targets_batch = np.array([0, 1, 2, 3, 4])
    result_batch = _one_hot_encode(targets_batch, batch_size=5, num_classes=5)
    assert np.allclose(result_batch.sum(axis=1), np.ones(5)), 'Each row should sum to 1'

def test_unit_tensor_autograd():
    """Test Tensor autograd enhancement."""
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1
    z.backward()
    assert np.allclose(x.grad, [3.0]), f'Expected [3.0], got {x.grad}'
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0], [4.0]], requires_grad=True)
    c = a.matmul(b)
    c.backward()
    assert np.allclose(a.grad, [[3.0, 4.0]]), f'Expected [[3.0, 4.0]], got {a.grad}'
    assert np.allclose(b.grad, [[1.0], [2.0]]), f'Expected [[1.0], [2.0]], got {b.grad}'
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()
    assert np.allclose(x.grad, [2.0, 2.0]), f'Expected [2.0, 2.0], got {x.grad}'

def test_module():
    test_unit_stable_softmax()
    test_unit_one_hot_encode()
    test_unit_function_classes()
    test_unit_tensor_autograd()
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    W1 = Tensor([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]], requires_grad=True)
    b1 = Tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    h1 = x.matmul(W1) + b1
    assert h1.shape == (1, 3)
    assert h1.requires_grad == True
    W2 = Tensor([[0.1], [0.2], [0.3]], requires_grad=True)
    h2 = h1.matmul(W2)
    assert h2.shape == (1, 1)
    loss = h2 * h2
    loss.backward()
    assert x.grad is not None
    assert W1.grad is not None
    assert b1.grad is not None
    assert W2.grad is not None
    assert x.grad.shape == x.shape
    assert W1.grad.shape == W1.shape
    x = Tensor([2.0], requires_grad=True)
    y1 = x * 3
    y1.backward()
    first_grad = x.grad.copy()
    y2 = x * 5
    y2.backward()
    assert np.allclose(x.grad, first_grad + 5.0), 'Gradients should accumulate'
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    temp1 = a.matmul(b)
    temp2 = temp1 + a
    result = temp2 * b
    final = result.sum()
    final.backward()
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
