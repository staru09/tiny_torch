import random
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

import numpy as np

from src.tensor import Tensor


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        if len(tensors) == 0:
            raise ValueError("Must provide at least one tensor")

        self.tensors = tensors
        n = len(tensors[0].data)
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != n:
                raise ValueError(
                    f"Tensor size mismatch: tensor 0 has {n} samples, "
                    f"tensor {i} has {len(tensor.data)}"
                )

    def __len__(self) -> int:
        return len(self.tensors[0].data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_batch(batch)

    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        if len(batch) == 0:
            return ()

        num_tensors = len(batch[0])
        batched = []
        for tensor_idx in range(num_tensors):
            stacked = np.stack([sample[tensor_idx].data for sample in batch], axis=0)
            batched.append(Tensor(stacked))
        return tuple(batched)
