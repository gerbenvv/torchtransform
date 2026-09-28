from unittest import TestCase

import torch

from torchtransform.color import (
    AutoContrast,
    Brightness,
    ChannelDropout,
    ChannelShuffle,
    ColorJitter,
    ColorMatrix,
    Contrast,
    Grayscale,
    Hue,
    Invert,
    Normalize,
    RGBShift,
    Saturation,
    Sepia,
)
from torchtransform.functional import rgb_to_grayscale
from torchtransform.geometric import Rotate
from torchtransform.targets import Mask
from torchtransform.tests.helpers import DEVICES
from torchtransform.transform import Compose, Inverse


def every_color_transform() -> tuple:
    """Returns one of every color transform."""

    return (
        Brightness(),
        Brightness(per_channel=True),
        Contrast(),
        Saturation(),
        Hue(),
        ColorJitter(0.3, 0.3, 0.3, 0.1),
        Grayscale(),
        Invert(),
        RGBShift(),
        ChannelShuffle(),
        ChannelDropout(),
        Sepia((0.2, 0.8)),
        AutoContrast(),
    )


class ColorTest(TestCase):
    def test_images_keep_their_shape_and_range(self) -> None:
        torch.manual_seed(0)

        for device in DEVICES:
            images = torch.rand(4, 3, 8, 8, device=device)

            for transform in every_color_transform():
                output = transform(images, seed=1)

                name = type(transform).__name__
                self.assertEqual(output.shape, images.shape, name)
                self.assertEqual(output.device, images.device, name)
                self.assertTrue(bool((output >= 0).all() and (output <= 1).all()), name)

    def test_masks_are_left_alone(self) -> None:
        masks = torch.rand(2, 3, 8, 8)

        for transform in every_color_transform():
            self.assertTrue(torch.equal(transform(Mask(masks, mode="bilinear")), masks))

    def test_a_run_is_fused_into_one_pass(self) -> None:
        images = torch.rand(4, 3, 8, 8) * 0.5 + 0.25
        brightness, contrast, saturation = (
            Brightness((0.9, 1.1)),
            Contrast((0.9, 1.1)),
            Saturation((0.9, 1.1)),
        )

        fused = Compose(brightness, contrast, saturation)(images, seed=2)
        separate = saturation(contrast(brightness(images, seed=2), seed=2), seed=2)

        # Without values leaving [0, 1] on the way, fusing changes nothing.
        torch.testing.assert_close(fused, separate)

    def test_contrast_blends_with_the_mean_after_what_came_before(self) -> None:
        images = torch.rand(2, 3, 8, 8) * 0.5

        output = Compose(Brightness(2.0), Contrast(0.0))(images)

        expected = rgb_to_grayscale(images * 2).mean(dim=(1, 2, 3))
        torch.testing.assert_close(output[:, 0, 0, 0], expected)

    def test_contrast_after_geometry_uses_the_transformed_image(self) -> None:
        images = torch.rand(2, 3, 16, 16)

        output = Compose(Rotate(45), Contrast(0.0))(images)
        expected = rgb_to_grayscale(Rotate(45)(images)).mean(dim=(1, 2, 3))

        torch.testing.assert_close(output[:, 0, 0, 0], expected)

    def test_known_values(self) -> None:
        images = torch.tensor((0.2, 0.4, 0.6)).view(1, 3, 1, 1)

        torch.testing.assert_close(Invert()(images).flatten(), torch.tensor((0.8, 0.6, 0.4)))
        torch.testing.assert_close(Brightness(0.5)(images).flatten(), torch.tensor((0.1, 0.2, 0.3)))
        torch.testing.assert_close(Saturation(1.0)(images), images)
        torch.testing.assert_close(Hue(0.0)(images), images)

        gray = 0.299 * 0.2 + 0.587 * 0.4 + 0.114 * 0.6
        torch.testing.assert_close(Grayscale()(images).flatten(), torch.full((3,), gray))

        # Two half turns of the hue are no turn at all, up to the rounding of the CSS matrix.
        half = Hue((0.5, 0.5))
        torch.testing.assert_close(half(half(images)), images, atol=2e-3, rtol=0)

        sums = ChannelShuffle()(images).flatten().sort().values
        torch.testing.assert_close(sums, images.flatten())

    def test_hue_keeps_the_luma(self) -> None:
        images = torch.rand(4, 3, 8, 8) * 0.5 + 0.25

        output = Hue((-0.05, 0.05))(images)
        torch.testing.assert_close(
            rgb_to_grayscale(output), rgb_to_grayscale(images), atol=0.02, rtol=0
        )

    def test_normalize_and_its_inverse(self) -> None:
        images = torch.rand(2, 3, 8, 8)
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        output = normalize(images)
        expected = (images - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)) / torch.tensor(
            (0.229, 0.224, 0.225)
        ).view(1, 3, 1, 1)

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(Inverse(normalize)(output), images)

    def test_autocontrast_stretches_every_channel(self) -> None:
        images = torch.rand(2, 3, 8, 8) * 0.5 + 0.2

        output = AutoContrast()(images)

        torch.testing.assert_close(output.amin(dim=(2, 3)), torch.zeros(2, 3))
        torch.testing.assert_close(output.amax(dim=(2, 3)), torch.ones(2, 3))

    def test_other_channel_counts(self) -> None:
        images = torch.rand(2, 1, 8, 8)

        self.assertEqual(Compose(Brightness(), Contrast(), Invert())(images).shape, images.shape)
        self.assertEqual(ColorJitter(brightness=0.2, contrast=0.2)(images).shape, images.shape)

        for transform in (Saturation(), Hue(), Grayscale(), Sepia(), ColorJitter(saturation=0.2)):
            with self.assertRaises(ValueError):
                transform(images)

    def test_color_matrix(self) -> None:
        images = torch.rand(2, 3, 4, 4)
        swap = torch.tensor(((0.0, 1, 0), (1, 0, 0), (0, 0, 1)))

        output = ColorMatrix(swap)(images)
        torch.testing.assert_close(output, images[:, (1, 0, 2)])
