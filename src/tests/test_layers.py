import numpy as np
import pytest
from src.activations import ReLU
from src.layers import Dropout, Linear
from src.tensor import Tensor
INIT_SCALE_FACTOR = 1.0
rng = np.random.default_rng(7)
requires_dropout_helpers = pytest.mark.skipif(not hasattr(Dropout(0.5), '_should_apply_dropout'), reason='Dropout helper methods not implemented in src.layers')

def test_unit_linear_layer():
    """Test Linear layer implementation."""
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)
    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(INIT_SCALE_FACTOR / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, f'Weight std {weight_std} not close to expected {expected_std}'
    assert np.allclose(layer.bias.data, 0), 'Bias should be initialized to zeros'
    x = Tensor(rng.standard_normal((32, 784)))
    y = layer.forward(x)
    assert y.shape == (32, 256), f'Expected shape (32, 256), got {y.shape}'
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1
    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias

def test_unit_edge_cases_linear():
    """Test Linear layer edge cases."""
    layer = Linear(10, 5)
    x_2d = Tensor(rng.standard_normal((1, 10)))
    y = layer.forward(x_2d)
    assert y.shape == (1, 5), 'Should handle single sample'
    x_empty = Tensor(rng.standard_normal((0, 10)))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0, 5), 'Should handle empty batch'
    layer_large = Linear(10, 5)
    layer_large.weight.data = np.ones((10, 5)) * 100
    x = Tensor(np.ones((1, 10)))
    y = layer_large.forward(x)
    assert not np.any(np.isnan(y.data)), 'Should not produce NaN with large weights'
    assert not np.any(np.isinf(y.data)), 'Should not produce Inf with large weights'
    layer_no_bias = Linear(10, 5, bias=False)
    x = Tensor(rng.standard_normal((4, 10)))
    y = layer_no_bias.forward(x)
    assert y.shape == (4, 5), 'Should work without bias'

def test_unit_parameter_collection_linear():
    """Test Linear layer parameter collection."""
    layer = Linear(10, 5)
    params = layer.parameters()
    assert len(params) == 2, 'Should return 2 parameters (weight and bias)'
    assert params[0].shape == (10, 5), 'First param should be weight'
    assert params[1].shape == (5,), 'Second param should be bias'
    layer_no_bias = Linear(10, 5, bias=False)
    params_no_bias = layer_no_bias.parameters()
    assert len(params_no_bias) == 1, 'Should return 1 parameter (weight only)'

@requires_dropout_helpers
def test_unit_should_apply_dropout():
    """Test _should_apply_dropout decision logic."""
    d = Dropout(0.5)
    assert d._should_apply_dropout(training=True) is True, 'Dropout(0.5) should apply during training'
    assert d._should_apply_dropout(training=False) is False, 'Dropout should not apply during inference'
    d_zero = Dropout(0.0)
    assert d_zero._should_apply_dropout(training=True) is False, 'Dropout(0.0) should never apply (no neurons to drop)'
    d_full = Dropout(1.0)
    assert d_full._should_apply_dropout(training=True) is True, 'Dropout(1.0) should apply during training'
    assert d_full._should_apply_dropout(training=False) is False, 'Even Dropout(1.0) should not apply during inference'

@requires_dropout_helpers
def test_unit_generate_dropout_mask():
    """Test _generate_dropout_mask output properties."""
    d = Dropout(0.5)
    rng = np.random.default_rng(7)
    mask = d._generate_dropout_mask((1000,))
    assert mask.shape == (1000,), f'Expected shape (1000,), got {mask.shape}'
    unique_vals = set(np.unique(mask.data))
    assert unique_vals <= {0.0, 2.0}, f'Mask values should be {{0.0, 2.0}}, got {unique_vals}'
    non_zero = np.count_nonzero(mask.data)
    std_err = np.sqrt(1000 * 0.5 * 0.5)
    assert 500 - 3 * std_err < non_zero < 500 + 3 * std_err, f'Expected ~500 survivors, got {non_zero}'
    d2 = Dropout(0.3)
    rng = np.random.default_rng(7)
    mask2 = d2._generate_dropout_mask((2000,))
    expected_scale = 1.0 / 0.7
    non_zero_vals = mask2.data[mask2.data != 0.0]
    assert np.allclose(non_zero_vals, expected_scale), f'Surviving values should be {expected_scale:.4f}, got {np.unique(non_zero_vals)}'
    survival_rate = np.count_nonzero(mask2.data) / 2000
    assert 0.6 < survival_rate < 0.8, f'Expected ~70% survival for p=0.3, got {survival_rate:.1%}'

def test_unit_dropout_layer():
    """Test Dropout layer implementation."""
    dropout = Dropout(0.5)
    assert dropout.p == 0.5
    x = Tensor([1, 2, 3, 4])
    y_inference = dropout.forward(x, training=False)
    assert np.array_equal(x.data, y_inference.data), 'Inference should pass through unchanged'
    dropout_zero = Dropout(0.0)
    y_zero = dropout_zero.forward(x, training=True)
    assert np.array_equal(x.data, y_zero.data), 'Zero dropout should pass through unchanged'
    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x, training=True)
    assert np.allclose(y_full.data, 0), 'Full dropout should zero everything'
    rng = np.random.default_rng(7)
    x_large = Tensor(np.ones((1000,)))
    y_train = dropout.forward(x_large, training=True)
    non_zero_count = np.count_nonzero(y_train.data)
    expected = 500
    std_error = np.sqrt(1000 * 0.5 * 0.5)
    lower_bound = expected - 3 * std_error
    upper_bound = expected + 3 * std_error
    assert lower_bound < non_zero_count < upper_bound, f'Expected {expected}±{3 * std_error:.0f} survivors, got {non_zero_count}'
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0
    assert np.allclose(surviving_values, expected_value), f'Surviving values should be {expected_value}'
    params = dropout.parameters()
    assert len(params) == 0, 'Dropout should have no parameters'
    try:
        Dropout(-0.1)
        assert False, 'Should raise ValueError for negative probability'
    except ValueError:
        pass
    try:
        Dropout(1.1)
        assert False, 'Should raise ValueError for probability > 1'
    except ValueError:
        pass

def test_module():
    test_unit_linear_layer()
    test_unit_edge_cases_linear()
    test_unit_parameter_collection_linear()
    if hasattr(Dropout(0.5), '_should_apply_dropout'):
        test_unit_should_apply_dropout()
        test_unit_generate_dropout_mask()
    test_unit_dropout_layer()
    ReLU_class = ReLU
    layer1 = Linear(784, 128)
    activation1 = ReLU_class()
    dropout1 = Dropout(0.5)
    layer2 = Linear(128, 64)
    activation2 = ReLU_class()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64, 10)
    batch_size = 16
    x = Tensor(rng.standard_normal((batch_size, 784)))
    x = layer1.forward(x)
    x = activation1.forward(x)
    x = dropout1.forward(x)
    x = layer2.forward(x)
    x = activation2.forward(x)
    x = dropout2.forward(x)
    output = layer3.forward(x)
    assert output.shape == (batch_size, 10), f'Expected output shape ({batch_size}, 10), got {output.shape}'
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6
    assert len(all_params) == expected_params, f'Expected {expected_params} parameters, got {len(all_params)}'
    test_x = Tensor(rng.standard_normal((4, 784)))
    dropout_test = Dropout(0.5)
    train_output = dropout_test.forward(test_x, training=True)
    infer_output = dropout_test.forward(test_x, training=False)
    assert np.array_equal(test_x.data, infer_output.data), 'Inference mode should pass through unchanged'
