from __future__ import annotations
import torch
import math
from typing import NamedTuple

class DataSample(NamedTuple):
    n: int
    features: torch.Tensor
    label: int


    @staticmethod
    def calculate_label(n: int) -> int:
        return [1, 3, 5, 15].index(math.gcd(n, 15))


    @staticmethod
    def binary_digits(n: int, num_digits: int = 10) -> torch.Tensor:
        digits = []
        for _ in range(num_digits):
            digits.append(float(n % 2))
            n = n // 2
        return torch.tensor(digits)


    @staticmethod
    def create(n: int) -> DataSample:
        return DataSample(n, DataSample.binary_digits(n), DataSample.calculate_label(n))