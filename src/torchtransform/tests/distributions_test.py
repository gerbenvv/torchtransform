from unittest import TestCase

import torch

from torchtransform.distributions import (
    Choice,
    Constant,
    IntUniform,
    LogUniform,
    Normal,
    Uniform,
    symmetric,
    symmetric_log,
    to_distribution,
)


class DistributionsTest(TestCase):
    def test_parameters_become_distributions(self) -> None:
        self.assertIsInstance(to_distribution(2.0), Constant)
        self.assertIsInstance(to_distribution((1.0, 2.0)), Uniform)
        self.assertIsInstance(to_distribution((1.0, 2.0), log=True), LogUniform)
        self.assertIsInstance(to_distribution((1, 3), integer=True), IntUniform)
        self.assertIsInstance(to_distribution([1.0, 5.0]), Choice)
        self.assertIsInstance(to_distribution((2.0, 2.0)), Constant)

        normal = Normal(0.0, 1.0)
        self.assertIs(to_distribution(normal), normal)

        with self.assertRaises(TypeError):
            to_distribution("large")

        with self.assertRaises(ValueError):
            to_distribution((1.0, 2.0, 3.0))

    def test_draws_stay_in_range(self) -> None:
        generator = torch.Generator().manual_seed(0)

        for distribution in (
            Uniform(2.0, 5.0),
            LogUniform(0.5, 2.0),
            IntUniform(1, 3),
            Choice([1.0, 4.0]),
        ):
            values = distribution.sample((1000,), generator)

            self.assertEqual(values.dtype, torch.float64)
            self.assertTrue(
                bool((values >= distribution.low).all() and (values <= distribution.high).all())
            )

        self.assertEqual(set(IntUniform(1, 3).sample((300,), generator).tolist()), {1.0, 2.0, 3.0})

    def test_log_uniform_is_symmetric_around_one(self) -> None:
        values = LogUniform(0.5, 2.0).sample((20000,), torch.Generator().manual_seed(0))

        self.assertAlmostEqual(float((values > 1).double().mean()), 0.5, delta=0.02)

    def test_symmetric_ranges(self) -> None:
        self.assertEqual(symmetric(10), (-10, 10))
        self.assertEqual(symmetric((0, 5)), (0, 5))
        self.assertEqual(symmetric_log(2.0), (0.5, 2.0))
        self.assertEqual(symmetric_log(0.5), (0.5, 2.0))

    def test_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            Uniform(2.0, 1.0)

        with self.assertRaises(ValueError):
            LogUniform(0.0, 1.0)

        with self.assertRaises(ValueError):
            Choice([])
