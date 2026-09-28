from unittest import TestCase

import torch

from torchtransform.canvas import Pad, RandomCrop
from torchtransform.geometric import Flip, QuarterTurn, Rotate, Symmetry, Transpose
from torchtransform.resampling import MatrixStep, Sampling
from torchtransform.targets import Image, Mask
from torchtransform.tests.helpers import DEVICES
from torchtransform.transform import Compose
from torchtransform.warps import ElasticWarp


def force_grid_sample(transform, *inputs, **targets):
    """Calls a transform with the exact paths turned off, so all goes through `grid_sample`."""

    original = Sampling._get_kind

    def get_kind(self):
        kind = original(self)
        return "affine" if kind in ("gather", "mixed", "identity") else kind

    Sampling._get_kind = get_kind
    try:
        return transform(*inputs, **targets)
    finally:
        Sampling._get_kind = original


class SamplingTest(TestCase):
    def test_the_kind_of_a_chain(self) -> None:
        identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)

        self.assertEqual(
            Sampling([MatrixStep(identity, (4, 4), (4, 4))], (4, 4), (4, 4), 2).kind, "identity"
        )

        flip = identity.clone()
        flip[:, 0, 0] = -1
        self.assertEqual(
            Sampling([MatrixStep(flip, (4, 4), (4, 4))], (4, 4), (4, 4), 2).kind, "gather"
        )

        half = identity.clone()
        half[:, 0, 2] = 0.5
        self.assertEqual(
            Sampling([MatrixStep(half, (4, 4), (4, 4))], (4, 4), (4, 4), 2).kind, "affine"
        )

        mixed = identity.clone()
        mixed[1, 0, 2] = 0.5
        self.assertEqual(
            Sampling([MatrixStep(mixed, (4, 4), (4, 4))], (4, 4), (4, 4), 2).kind, "mixed"
        )

    def test_gathering_matches_grid_sample(self) -> None:
        torch.manual_seed(0)

        for device in DEVICES:
            images = torch.rand(8, 3, 15, 17, device=device)
            labels = torch.randint(0, 5, (8, 1, 15, 17), device=device)

            transform = Compose(Symmetry(), Transpose(), RandomCrop((13, 12)), Pad((2, 3)))

            for padding in ("zeros", "border", "reflection", 0.5):
                targets = dict(image=Image(images, padding=padding), mask=Mask(labels, padding=255))

                exact = transform(**targets, seed=3)
                sampled = force_grid_sample(transform, **targets, seed=3)

                torch.testing.assert_close(exact["image"], sampled["image"], atol=1e-5, rtol=0)
                self.assertTrue(torch.equal(exact["mask"], sampled["mask"]), padding)

    def test_gathering_in_3d(self) -> None:
        torch.manual_seed(0)
        volumes = torch.rand(4, 2, 6, 7, 8)

        transform = Compose(Symmetry(), QuarterTurn(axes=("x", "z")), RandomCrop((5, 6, 6)))

        exact = transform(volumes, seed=1)
        sampled = force_grid_sample(transform, volumes, seed=1)

        torch.testing.assert_close(exact, sampled, atol=1e-5, rtol=0)

    def test_a_uniform_crop_is_a_view(self) -> None:
        images = torch.rand(2, 3, 10, 10)

        output = Compose(Flip("x", p=0.0), RandomCrop(10))(images)
        self.assertEqual(output.data_ptr(), images.data_ptr())

    def test_mixed_batches_keep_exact_elements_exact(self) -> None:
        torch.manual_seed(0)
        images = torch.rand(64, 3, 16, 16)

        transform = Compose(
            Rotate((-30, 30), p=0.5).manual_seed(1),
            ElasticWarp(p=0.5).manual_seed(2),
            Flip("x").manual_seed(3),
        )
        output = transform(images, seed=2)

        # The elements that were neither rotated nor warped are only flipped, so exactly.
        exact = [
            torch.equal(output[i], images[i]) or torch.equal(output[i], images[i].flip(-1))
            for i in range(len(images))
        ]
        self.assertTrue(5 < sum(exact) < 40)

    def test_bicubic_interpolation(self) -> None:
        images = torch.rand(2, 3, 16, 16)

        output = Rotate(15)(Image(images, mode="bicubic"))
        self.assertEqual(output.shape, images.shape)

    def test_half_precision_on_the_gpu(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("No GPU.")

        images = torch.rand(2, 3, 16, 16, device="cuda", dtype=torch.float16)

        output = Compose(Rotate(15), ElasticWarp())(images)
        self.assertEqual(output.dtype, torch.float16)
