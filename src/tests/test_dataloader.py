import random

import numpy as np

from src.dataloader import DataLoader, Dataset, TensorDataset
from src.tensor import Tensor


def test_unit_dataset():
    try:
        Dataset()
        assert False, "Should not instantiate abstract Dataset"
    except TypeError:
        pass

    class TestDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return f"item_{idx}"

    dataset = TestDataset(10)
    assert len(dataset) == 10
    assert dataset[0] == "item_0"
    assert dataset[9] == "item_9"


def test_unit_tensordataset():
    features = Tensor([[1, 2], [3, 4], [5, 6]])
    labels = Tensor([0, 1, 0])
    dataset = TensorDataset(features, labels)

    assert len(dataset) == 3

    sample = dataset[0]
    assert len(sample) == 2
    assert np.array_equal(sample[0].data, [1, 2])
    assert sample[1].data == 0

    sample = dataset[1]
    assert np.array_equal(sample[1].data, 1)

    try:
        dataset[10]
        assert False
    except IndexError:
        pass

    try:
        TensorDataset(Tensor([[1, 2], [3, 4]]), Tensor([0, 1, 0]))
        assert False
    except ValueError:
        pass


def test_unit_dataloader():
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    labels = Tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(features, labels)

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    assert len(loader) == 3

    batches = list(loader)
    assert len(batches) == 3

    batch_features, batch_labels = batches[0]
    assert batch_features.data.shape == (2, 2)
    assert batch_labels.data.shape == (2,)

    batch_features, batch_labels = batches[2]
    assert batch_features.data.shape == (1, 2)
    assert batch_labels.data.shape == (1,)

    assert np.array_equal(batches[0][0].data[0], [1, 2])
    assert batches[0][1].data[0] == 0

    loader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
    loader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)
    batch_shuffle = list(loader_shuffle)[0]
    batch_no_shuffle = list(loader_no_shuffle)[0]

    shuffle_features = {tuple(row) for row in batch_shuffle[0].data}
    no_shuffle_features = {tuple(row) for row in batch_no_shuffle[0].data}
    expected_features = {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)}
    assert shuffle_features == expected_features
    assert no_shuffle_features == expected_features


def test_unit_dataloader_deterministic():
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    labels = Tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)

    random.seed(42)
    batches1 = list(DataLoader(dataset, batch_size=2, shuffle=True))

    random.seed(42)
    batches2 = list(DataLoader(dataset, batch_size=2, shuffle=True))

    for batch1, batch2 in zip(batches1, batches2):
        assert np.array_equal(batch1[0].data, batch2[0].data)
        assert np.array_equal(batch1[1].data, batch2[1].data)

    random.seed(42)
    batches3 = list(DataLoader(dataset, batch_size=2, shuffle=True))

    random.seed(123)
    batches4 = list(DataLoader(dataset, batch_size=2, shuffle=True))

    different = any(
        not np.array_equal(b3[0].data, b4[0].data)
        for b3, b4 in zip(batches3, batches4)
    )
    assert different


def test_module():
    test_unit_dataset()
    test_unit_tensordataset()
    test_unit_dataloader()
    test_unit_dataloader_deterministic()

    features = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    labels = Tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=True)

    batch_count = 0
    for batch_x, batch_y in loader:
        assert batch_x.shape[0] == batch_y.shape[0]
        assert batch_x.shape[1] == 2
        batch_count += 1
    assert batch_count == 2
