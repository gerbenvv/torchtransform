from unittest import TestCase

import torch

from torchtransform.photometric import (
    BackgroundGradient,
    BackgroundIrregularities,
    ChromaticAberration,
    Clamp,
    Defocus,
    Downscale,
    Equalize,
    Erasing,
    Gamma,
    GaussianBlur,
    GaussianNoise,
    JpegCompression,
    MotionBlur,
    Pixelate,
    PoissonNoise,
    Posterize,
    SaltAndPepperNoise,
    Sharpen,
    Solarize,
    SpeckleNoise,
    Vignette,
)
from torchtransform.targets import Image, Mask
from torchtransform.tests.helpers import DEVICES
from torchtransform.transform import PixelTransform


def every_transform() -> tuple[PixelTransform, ...]:
    """Returns one of every photometric transform, with parameters that change the image."""

    return (
        Gamma((0.5, 2.0)),
        Solarize((0.3, 0.7)),
        Posterize((2, 5)),
        Equalize(),
        Sharpen(),
        GaussianBlur((0.5, 2.0)),
        MotionBlur(),
        Defocus(),
        GaussianNoise((0.05, 0.1)),
        GaussianNoise((0.05, 0.1), per_channel=False),
        PoissonNoise(),
        SpeckleNoise((0.05, 0.2)),
        SaltAndPepperNoise((0.05, 0.1)),
        JpegCompression((10, 50)),
        JpegCompression((10, 50), subsampling=False),
        Downscale(),
        Downscale(mode="nearest"),
        Pixelate(),
        Erasing(),
        Erasing(fill="noise"),
        Erasing(fill="mean"),
        Vignette(),
        BackgroundIrregularities(),
        BackgroundGradient(),
        Clamp(0.2, 0.8),
    )


def is_2d_only(transform: PixelTransform) -> bool:
    """Returns whether a transform is only available in 2D."""

    return isinstance(transform, (MotionBlur, Defocus, JpegCompression))


class PhotometricTest(TestCase):
    def test_shape_dtype_and_range_are_kept(self) -> None:
        for device in DEVICES:
            for transform in every_transform():
                for dtype in (torch.float32, torch.float64):
                    torch.manual_seed(0)
                    images = torch.rand(3, 3, 21, 34, device=device, dtype=dtype)

                    output = transform(images, seed=1)

                    name = f"{type(transform).__name__} on {device} in {dtype}"
                    self.assertEqual(output.shape, images.shape, name)
                    self.assertEqual(output.dtype, dtype, name)
                    self.assertEqual(output.device, images.device, name)
                    self.assertTrue(bool(torch.isfinite(output).all()), name)
                    self.assertGreaterEqual(output.min().item(), 0.0, name)
                    self.assertLessEqual(output.max().item(), 1.0, name)
                    self.assertFalse(torch.equal(output, images), name)

    def test_uint8_images_come_back_as_uint8(self) -> None:
        images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)

        for transform in every_transform():
            output = transform(images, seed=1)

            self.assertEqual(output.dtype, torch.uint8, type(transform).__name__)

    def test_single_channel_images(self) -> None:
        images = torch.rand(2, 1, 24, 24)

        for transform in every_transform():
            output = transform(images, seed=1)

            self.assertEqual(output.shape, images.shape, type(transform).__name__)

    def test_3d_images(self) -> None:
        for transform in every_transform():
            images = torch.rand(2, 2, 9, 12, 10)

            if is_2d_only(transform):
                with self.assertRaises(ValueError):
                    transform(images, seed=1)

                continue

            output = transform(images, seed=1)

            name = type(transform).__name__
            self.assertEqual(output.shape, images.shape, name)
            self.assertTrue(bool(torch.isfinite(output).all()), name)
            self.assertGreaterEqual(output.min().item(), 0.0, name)
            self.assertLessEqual(output.max().item(), 1.0, name)

    def test_deterministic_under_a_seed(self) -> None:
        images = torch.rand(3, 3, 20, 20)

        for transform in every_transform():
            first = transform(images, seed=7)
            second = transform(images, seed=7)
            other = transform(images, seed=8)

            name = type(transform).__name__
            self.assertTrue(torch.equal(first, second), name)

            if not isinstance(transform, (Equalize, Clamp)):
                self.assertFalse(torch.equal(first, other), name)

    def test_deterministic_under_the_global_seed(self) -> None:
        images = torch.rand(2, 3, 20, 20)

        for transform in every_transform():
            torch.manual_seed(3)
            first = transform(images)

            torch.manual_seed(3)
            second = transform(images)

            self.assertTrue(torch.equal(first, second), type(transform).__name__)

    def test_elements_draw_their_own_parameters(self) -> None:
        # The same image in every element of the batch comes out differently per element.
        images = torch.rand(1, 3, 24, 24).expand(4, -1, -1, -1).contiguous()

        for transform in every_transform():
            if isinstance(transform, (Equalize, Clamp)):
                continue

            output = transform(images, seed=2)

            self.assertFalse(torch.equal(output[0], output[1]), type(transform).__name__)

    def test_elements_it_does_not_apply_to_are_untouched(self) -> None:
        images = torch.rand(16, 3, 16, 16)

        for transform in every_transform():
            transform.p = 0.5

            output = transform(images, seed=4)

            untouched = [bool(torch.equal(output[i], images[i])) for i in range(16)]
            name = type(transform).__name__
            self.assertTrue(any(untouched), name)
            self.assertFalse(all(untouched), name)

    def test_an_element_does_not_depend_on_the_others(self) -> None:
        # Elements that are applied to get the same parameters whichever others are.
        images = torch.rand(8, 3, 16, 16)

        for transform in (Gamma((0.5, 2.0)), Posterize((2, 5)), Solarize((0.3, 0.7)), Vignette()):
            full = transform(images, seed=5)

            transform.p = 0.5
            partial = transform(images, seed=5)

            for i in range(8):
                if not torch.equal(partial[i], images[i]):
                    self.assertTrue(torch.equal(partial[i], full[i]), type(transform).__name__)

    def test_images_are_not_changed_in_place(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        original = images.clone()

        for transform in every_transform():
            transform(images, seed=1)

            self.assertTrue(torch.equal(images, original), type(transform).__name__)

    def test_masks_are_left_alone(self) -> None:
        masks = torch.randint(0, 5, (2, 1, 16, 16))

        for transform in every_transform():
            outputs = transform(image=Image(torch.rand(2, 3, 16, 16)), mask=Mask(masks), seed=1)

            self.assertTrue(torch.equal(outputs["mask"], masks), type(transform).__name__)

    def test_every_image_gets_the_same_noise(self) -> None:
        images = torch.rand(2, 3, 16, 16)

        for transform in (GaussianNoise(), SaltAndPepperNoise(), Erasing(fill="noise")):
            outputs = transform(first=Image(images), second=Image(images.clone()), seed=1)

            self.assertTrue(torch.equal(outputs["first"], outputs["second"]))

    def test_identity_settings_leave_images_alone(self) -> None:
        images = torch.rand(2, 3, 16, 16, dtype=torch.float64)

        for transform in (
            Gamma(1.0),
            Solarize(1.5),
            Sharpen(amount=0.0),
            GaussianNoise(0.0),
            SpeckleNoise(0.0),
            SaltAndPepperNoise(0.0),
            Erasing(count=0),
            Vignette(strength=0.0),
            Downscale(1.0),
            Pixelate(1.0),
            MotionBlur(length=1.0),
            Defocus(radius=0.0),
        ):
            output = transform(images, seed=1)

            torch.testing.assert_close(output, images, msg=type(transform).__name__)

    def test_blurs_keep_a_constant_image(self) -> None:
        images = torch.full((2, 3, 20, 20), 0.6)

        for transform in (GaussianBlur((0.5, 3.0)), MotionBlur((3.0, 11.0)), Defocus(), Sharpen()):
            output = transform(images, seed=1)

            torch.testing.assert_close(output, images, msg=type(transform).__name__)

    def test_motion_blur_blurs_along_its_direction(self) -> None:
        images = torch.zeros(1, 1, 21, 21)
        images[0, 0, 10, 10] = 1

        output = MotionBlur(length=7.0, angle=0.0)(images, seed=1)

        # A horizontal line through the dot, and nothing above or below it.
        self.assertGreater(output[0, 0, 10, 7].item(), 0.05)
        self.assertGreater(output[0, 0, 10, 13].item(), 0.05)
        self.assertLess(output[0, 0, 7, 10].item(), 1e-6)

    def test_posterize_quantizes(self) -> None:
        images = torch.rand(2, 3, 16, 16)

        output = Posterize(3)(images, seed=1)

        levels = torch.unique(torch.round(output * 255))
        self.assertLessEqual(len(levels), 8)
        self.assertTrue(bool((levels % 32 == 0).all()))

    def test_equalize_flattens_the_histogram(self) -> None:
        # A dark image with a narrow histogram spreads over the whole range.
        images = 0.2 + 0.1 * torch.rand(2, 1, 64, 64)

        output = Equalize()(images, seed=1)

        self.assertLess(output.min().item(), 0.05)
        self.assertGreater(output.max().item(), 0.95)

        histogram = torch.histc(output[0], bins=4, min=0, max=1)
        self.assertLess((histogram.max() / histogram.min()).item(), 1.5)

    def test_equalize_leaves_a_flat_channel_alone(self) -> None:
        images = torch.full((1, 3, 8, 8), 0.4)

        output = Equalize()(images, seed=1)

        torch.testing.assert_close(output, images)

    def test_solarize_inverts_above_the_threshold(self) -> None:
        images = torch.tensor((0.2, 0.8)).view(1, 1, 1, 2).expand(1, 1, 4, 2)

        output = Solarize(0.5)(images, seed=1)

        torch.testing.assert_close(output[0, 0, 0], torch.tensor((0.2, 0.2)))

    def test_jpeg_at_full_quality_is_nearly_lossless(self) -> None:
        torch.manual_seed(0)
        images = GaussianBlur(1.0)(torch.rand(2, 3, 32, 40), seed=1)

        lossless = JpegCompression(100, subsampling=False)(images, seed=1)
        lossy = JpegCompression(5)(images, seed=1)

        self.assertLess((lossless - images).abs().max().item(), 3 / 255)
        self.assertGreater(
            (lossy - images).abs().mean().item(), 5 * (lossless - images).abs().mean().item()
        )

    def test_jpeg_works_on_sizes_that_are_not_multiples_of_eight(self) -> None:
        images = torch.rand(1, 3, 13, 27)

        output = JpegCompression(50)(images, seed=1)

        self.assertEqual(output.shape, images.shape)

    def test_pixelate_makes_flat_cells(self) -> None:
        images = torch.rand(1, 3, 32, 32)

        output = Pixelate(8.0)(images, seed=1)

        # Every 8 by 8 cell is a single color.
        cells = output.view(1, 3, 4, 8, 4, 8)
        spread = cells.amax(dim=(3, 5)) - cells.amin(dim=(3, 5))
        self.assertLess(spread.max().item(), 1e-5)

    def test_downscale_removes_fine_detail(self) -> None:
        images = torch.zeros(1, 1, 32, 32)
        images[..., ::2, :] = 1

        output = Downscale(0.25)(images, seed=1)

        # Stripes of one pixel average out to gray.
        self.assertLess((output - 0.5).abs().max().item(), 0.1)

    def test_erasing_fills_boxes(self) -> None:
        images = torch.ones(2, 3, 32, 32)

        output = Erasing(count=2, size=0.25, fill=0.0)(images, seed=1)

        erased = (output == 0).all(dim=1)
        fractions = erased.flatten(1).float().mean(dim=1)
        self.assertTrue(bool((fractions > 0).all()))
        self.assertTrue(bool((fractions <= 2 * 0.25**2 + 0.05).all()))

    def test_vignette_darkens_the_corners_only(self) -> None:
        images = torch.ones(1, 3, 32, 32)

        output = Vignette(strength=0.5, radius=0.5)(images, seed=1)

        self.assertEqual(output[0, 0, 16, 16].item(), 1.0)
        self.assertLess(output[0, 0, 0, 0].item(), 0.7)

    def test_salt_and_pepper_changes_about_the_amount(self) -> None:
        images = torch.full((1, 3, 100, 100), 0.5)

        output = SaltAndPepperNoise(amount=0.2, salt=0.5)(images, seed=1)

        changed = (output != 0.5).all(dim=1).float().mean().item()
        white = (output == 1).all(dim=1).float().mean().item()
        self.assertAlmostEqual(changed, 0.2, delta=0.02)
        self.assertAlmostEqual(white, 0.1, delta=0.02)

    def test_backgrounds_are_restretched(self) -> None:
        images = torch.rand(2, 3, 24, 24)

        for transform in (BackgroundIrregularities(), BackgroundGradient()):
            output = transform(images, seed=1)

            self.assertLess(output.amin(dim=(1, 2, 3)).max().item(), 1e-5)
            self.assertGreater(output.amax(dim=(1, 2, 3)).min().item(), 0.99)

    def test_chromatic_aberration_fringes_away_from_the_center(self) -> None:
        images = torch.zeros(2, 3, 33, 33)
        images[:, :, 16, 16] = 1.0
        images[:, :, 2, 2] = 1.0

        output = ChromaticAberration((3.0, 3.0))(images)

        # The center stays sharp, and in a corner the red and blue channels part ways.
        torch.testing.assert_close(output[:, :, 16, 16], images[:, :, 16, 16])
        self.assertFalse(torch.equal(output[:, 0], output[:, 2]))
        torch.testing.assert_close(output[:, 1], images[:, 1])

        with self.assertRaises(ValueError):
            ChromaticAberration()(torch.rand(1, 1, 8, 8))
