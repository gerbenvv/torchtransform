"""Distributions that the random parameters of transforms are drawn from.

Every parameter of a transform takes a fixed number, a `(low, high)` tuple for a uniform range, a
list of choices, or any `Distribution`. Draws are float64 tensors on the CPU, one per batch element,
so every element of a batch is augmented independently.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

import torch


class Distribution(ABC):
    """A distribution of a scalar parameter."""

    @abstractmethod
    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        """Draws values.

        Args:
            shape: Shape of the draw.
            generator: CPU generator to draw with.

        Returns:
            A float64 tensor of the given shape on the CPU.
        """

        raise NotImplementedError()

    @property
    @abstractmethod
    def low(self) -> float:
        """The smallest value that can be drawn."""

        raise NotImplementedError()

    @property
    @abstractmethod
    def high(self) -> float:
        """The largest value that can be drawn."""

        raise NotImplementedError()


class Constant(Distribution):
    def __init__(self, value: float) -> None:
        """A distribution that always gives the same value.

        Args:
            value: The value.
        """

        self.value = float(value)

    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        return torch.full(shape, self.value, dtype=torch.float64)

    @property
    def low(self) -> float:
        return self.value

    @property
    def high(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Uniform(Distribution):
    def __init__(self, low: float, high: float) -> None:
        """A uniform distribution over `[low, high)`.

        Args:
            low: Lower bound.
            high: Upper bound.
        """

        if high < low:
            raise ValueError(f"The upper bound {high} is below the lower bound {low}.")

        self._low = float(low)
        self._high = float(high)

    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        values = torch.rand(shape, dtype=torch.float64, generator=generator)

        return values * (self._high - self._low) + self._low

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def __repr__(self) -> str:
        return f"Uniform({self._low}, {self._high})"


class LogUniform(Distribution):
    def __init__(self, low: float, high: float) -> None:
        """A distribution that is uniform in the logarithm over `[low, high)`.

        This is the natural distribution of a scale: zooming in by two is as likely as zooming out
        by two.

        Args:
            low: Lower bound, above zero.
            high: Upper bound.
        """

        if low <= 0:
            raise ValueError("The lower bound of a log-uniform distribution must be above zero.")

        if high < low:
            raise ValueError(f"The upper bound {high} is below the lower bound {low}.")

        self._low = float(low)
        self._high = float(high)

    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        values = torch.rand(shape, dtype=torch.float64, generator=generator)
        low, high = math.log(self._low), math.log(self._high)

        return torch.exp(values * (high - low) + low)

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def __repr__(self) -> str:
        return f"LogUniform({self._low}, {self._high})"


class IntUniform(Distribution):
    def __init__(self, low: int, high: int) -> None:
        """A uniform distribution over the integers from `low` to `high`, both included.

        Args:
            low: Lower bound.
            high: Upper bound, itself a possible outcome.
        """

        if high < low:
            raise ValueError(f"The upper bound {high} is below the lower bound {low}.")

        self._low = int(low)
        self._high = int(high)

    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        values = torch.randint(self._low, self._high + 1, shape, generator=generator)

        return values.to(dtype=torch.float64)

    @property
    def low(self) -> float:
        return float(self._low)

    @property
    def high(self) -> float:
        return float(self._high)

    def __repr__(self) -> str:
        return f"IntUniform({self._low}, {self._high})"


class Normal(Distribution):
    def __init__(
        self, mean: float, std: float, low: float = -math.inf, high: float = math.inf
    ) -> None:
        """A normal distribution, optionally truncated by clamping.

        Args:
            mean: Mean.
            std: Standard deviation.
            low: Values below are clamped to this.
            high: Values above are clamped to this.
        """

        self._mean = float(mean)
        self._std = float(std)
        self._low = float(low)
        self._high = float(high)

    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        values = (
            torch.randn(shape, dtype=torch.float64, generator=generator) * self._std + self._mean
        )

        return values.clamp(self._low, self._high)

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def __repr__(self) -> str:
        return f"Normal({self._mean}, {self._std})"


class Choice(Distribution):
    def __init__(self, values: Sequence[float], weights: Sequence[float] | None = None) -> None:
        """A distribution over a set of values.

        Args:
            values: The values.
            weights: Relative probability of every value. Equal if not given.
        """

        if len(values) == 0:
            raise ValueError("A choice needs at least one value.")

        if weights is not None and len(weights) != len(values):
            raise ValueError("A choice needs as many weights as values.")

        self._values = torch.tensor(tuple(values), dtype=torch.float64)
        self._weights = torch.tensor(
            tuple(weights) if weights is not None else (1.0,) * len(values), dtype=torch.float64
        )

    def sample(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        count = math.prod(shape)
        indices = torch.multinomial(self._weights, count, replacement=True, generator=generator)

        return self._values[indices].view(shape)

    @property
    def low(self) -> float:
        return float(self._values.min())

    @property
    def high(self) -> float:
        return float(self._values.max())

    def __repr__(self) -> str:
        return f"Choice({self._values.tolist()})"


# What a parameter of a transform accepts.
Parameter: TypeAlias = float | int | tuple[float, float] | list[float] | Distribution


def to_distribution(value: Parameter, log: bool = False, integer: bool = False) -> Distribution:
    """Turns a parameter into a distribution.

    Args:
        value: A number (always drawn), a `(low, high)` tuple (drawn uniformly), a list (one of its
            values, drawn uniformly), or a distribution (returned as-is).
        log: Whether a range is drawn uniformly in the logarithm, as a scale should be.
        integer: Whether a range is over integers, both bounds included.

    Returns:
        The distribution.
    """

    if isinstance(value, Distribution):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Constant(value)

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"A range must be a `(low, high)` tuple, not {value!r}.")

        low, high = value

        if integer:
            return IntUniform(int(low), int(high))

        if low == high:
            return Constant(low)

        return LogUniform(low, high) if log else Uniform(low, high)

    if isinstance(value, list):
        return Choice(value)

    raise TypeError(
        f"Expected a number, a `(low, high)` tuple, a list or a distribution, not {value!r}."
    )


def symmetric(value: Parameter) -> Parameter:
    """Turns a single number `a` into the range `(-a, a)`, and leaves anything else alone."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (-abs(value), abs(value))

    return value


def symmetric_log(value: Parameter) -> Parameter:
    """Turns a single factor `a` into the range `(1 / a, a)`, and leaves anything else alone."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = max(value, 1 / value)

        return (1 / value, value)

    return value
