from unittest import TestCase

import torch

from torchtransform.document import (
    MAX_SPECKLE_DENSITY,
    BitonalCopy,
    BleedThrough,
    ChannelMisalignment,
    DitherPattern,
    EdgeShadow,
    HalftoneDither,
    InkErosion,
    InkSpread,
    LineFragmentation,
    LowInkStreaks,
    PaperTint,
    RollerStreaks,
    _threshold_maps,
    bar_profiles,
    bayer_matrix,
    dither_tile,
    open_runs,
    tile_to,
)
from torchtransform.targets import Image, Mask, RotatedBoxes
from torchtransform.transform import PixelTransform


def every_augmentation(p: float = 1.0) -> tuple[PixelTransform, ...]:
    """Returns one of each augmentation, each turned up as far as its parameters allow."""

    return (
        BleedThrough(alpha=(0.18, 0.18), p=p),
        ChannelMisalignment(shift=(3, 3), mix=(0.25, 0.25), p=p),
        PaperTint(strength=(0.12, 0.12), p=p),
        HalftoneDither(pattern=DitherPattern.BAYER, strength=(0.9, 0.9), p=p),
        HalftoneDither(pattern=DitherPattern.CLUSTERED, strength=(0.9, 0.9), p=p),
        HalftoneDither(pattern=DitherPattern.LINE, strength=(0.9, 0.9), p=p),
        LowInkStreaks(alpha=(0.45, 0.45), count=(8, 8), p=p),
        RollerStreaks(strength=(0.25, 0.25), p=p),
        InkSpread(strength=(0.5, 0.5), p=p),
        InkErosion(strength=(0.4, 0.4), coverage=(0.35, 0.35), p=p),
        LineFragmentation(strength=(0.5, 0.5), rate=(0.25, 0.25), p=p),
        EdgeShadow(strength=(0.4, 0.4), p=p),
        BitonalCopy(stain_edges=(1, 2), p=p),
    )


def make_page(batch_size: int = 1) -> torch.Tensor:
    """Returns pages of white paper carrying a three pixel stroke and a one pixel hairline rule."""

    images = torch.ones(batch_size, 3, 64, 80)
    images[:, :, 20:23, 10:70] = 0.0
    images[:, :, 40:41, 5:75] = 0.0

    return images


class DocumentTest(TestCase):
    def test_boxes_are_never_touched(self) -> None:
        for augmentation in every_augmentation():
            boxes = torch.tensor([[[20.0, 20.0, 10.0, 6.0, 12.0], [40.0, 40.0, 70.0, 1.0, 0.0]]])

            outputs = augmentation(
                image=Image(torch.rand(1, 3, 64, 80)), boxes=RotatedBoxes(boxes), seed=0
            )

            # Every one of these is a pure pixel effect, so a box must come back as it was.
            torch.testing.assert_close(
                outputs["boxes"], boxes, atol=1e-5, rtol=0, msg=type(augmentation).__name__
            )

    def test_images_keep_their_shape_type_and_range(self) -> None:
        for augmentation in every_augmentation():
            images = torch.rand(2, 3, 64, 80)
            before = images.clone()

            output = augmentation(images, seed=0)

            name = type(augmentation).__name__
            self.assertTrue(torch.equal(images, before), name)
            self.assertEqual(output.shape, (2, 3, 64, 80), name)
            self.assertEqual(output.dtype, torch.float32, name)
            self.assertTrue(bool(torch.isfinite(output).all()), name)
            self.assertGreaterEqual(output.min().item(), 0.0, name)
            self.assertLessEqual(output.max().item(), 1.0, name)

    def test_nothing_happens_at_zero_probability(self) -> None:
        for augmentation in every_augmentation(p=0.0):
            images = torch.rand(2, 3, 32, 32)

            output = augmentation(images, seed=0)

            self.assertTrue(torch.equal(images, output), type(augmentation).__name__)

    def test_elements_it_does_not_apply_to_are_left_exactly_alone(self) -> None:
        for augmentation in every_augmentation(p=0.5):
            images = torch.rand(8, 3, 32, 40)

            output = augmentation(images, seed=3)

            unchanged = [bool(torch.equal(output[i], images[i])) for i in range(8)]
            name = type(augmentation).__name__

            self.assertTrue(any(unchanged), name)
            self.assertFalse(all(unchanged), name)

    def test_every_element_draws_its_own_parameters(self) -> None:
        for augmentation in every_augmentation():
            images = torch.rand(1, 3, 64, 80).expand(4, -1, -1, -1).contiguous()

            output = augmentation(images, seed=0)

            # The same page four times over comes out as more than one page.
            differing = any(not torch.equal(output[0], output[i]) for i in range(1, 4))
            self.assertTrue(differing, type(augmentation).__name__)

    def test_darkening_augmentations_only_ever_darken(self) -> None:
        for augmentation in (
            BleedThrough(alpha=(0.18, 0.18)),
            PaperTint(strength=(0.12, 0.12)),
            RollerStreaks(strength=(0.25, 0.25)),
            InkSpread(strength=(0.5, 0.5)),
            EdgeShadow(strength=(0.4, 0.4)),
        ):
            images = torch.rand(2, 3, 64, 80)

            output = augmentation(images, seed=0)

            # Darkening alone can never take ink away.
            self.assertLessEqual((output - images).max().item(), 1e-6, type(augmentation).__name__)

    def test_lightening_augmentations_only_ever_lighten(self) -> None:
        for augmentation in (
            LowInkStreaks(alpha=(0.45, 0.45), count=(8, 8)),
            InkErosion(strength=(0.4, 0.4)),
            LineFragmentation(strength=(0.5, 0.5)),
        ):
            images = torch.rand(2, 3, 64, 80)

            output = augmentation(images, seed=0)

            # Lightening alone can never invent ink that was not there.
            self.assertLessEqual((images - output).max().item(), 1e-6, type(augmentation).__name__)

    def test_a_stroke_survives_every_augmentation(self) -> None:
        for augmentation in every_augmentation():
            output = augmentation(make_page(20), seed=0)

            # However hard it is turned up, a three pixel stroke keeps a dark core on every page.
            worst = output[:, :, 20:23, 10:70].amin(dim=(1, 2, 3)).max().item()
            self.assertLess(worst, 0.35, type(augmentation).__name__)

    def test_a_hairline_rule_survives_the_effects_that_lighten_it(self) -> None:
        # These all lighten and can land on the same rule, so their caps have to hold together and
        # not only one at a time.
        erosion = InkErosion(strength=(0.4, 0.4), coverage=(0.35, 0.35))
        streaks = LowInkStreaks(alpha=(0.45, 0.45), coverage=(0.9, 0.9))
        fragmentation = LineFragmentation(strength=(0.5, 0.5), rate=(0.25, 0.25))

        images = make_page(30)
        for seed, augmentation in enumerate((erosion, streaks, fragmentation)):
            images = augmentation(images, seed=seed)

        # A rule is something a label can claim on its own, so it has to stay clearly visible.
        worst = images[:, :, 40:41, 5:75].mean(dim=(1, 2, 3)).max().item()
        self.assertLess(worst, 0.6)

    def test_augmentations_are_deterministic_under_a_seed(self) -> None:
        for augmentation in every_augmentation():
            images = torch.rand(2, 3, 48, 48)
            name = type(augmentation).__name__

            self.assertTrue(
                torch.equal(augmentation(images, seed=11), augmentation(images, seed=11)), name
            )

            # Without a seed, the call draws one from torch's global generator.
            torch.manual_seed(11)
            first = augmentation(images)
            torch.manual_seed(11)
            second = augmentation(images)

            self.assertTrue(torch.equal(first, second), name)

    def test_augmentations_cope_with_a_page_smaller_than_their_own_scale(self) -> None:
        # Several of these work at a coarser scale than the page, or place features by a period
        # that can be wider than the page is, so a small page is where the arithmetic gives out.
        for augmentation in every_augmentation():
            for size in (16, 24, 32):
                output = augmentation(torch.rand(2, 3, size, size), seed=0)

                self.assertTrue(
                    bool(torch.isfinite(output).all()), f"{type(augmentation).__name__} at {size}"
                )

    def test_augmentations_need_rgb_images_in_2d(self) -> None:
        with self.assertRaises(ValueError):
            PaperTint()(torch.rand(1, 1, 16, 16), seed=0)

        with self.assertRaises(ValueError):
            EdgeShadow()(torch.rand(1, 3, 4, 16, 16), seed=0)

    def test_bayer_matrix_holds_every_level_once(self) -> None:
        for order in range(1, 6):
            matrix = bayer_matrix(order)
            levels = (matrix.flatten() * matrix.numel() - 0.5).round().long()

            self.assertEqual(matrix.shape, (2**order, 2**order))
            self.assertEqual(len(torch.unique(levels)), matrix.numel())
            self.assertTrue(bool((matrix > 0).all() and (matrix < 1).all()))

    def test_bayer_matrix_rejects_a_negative_order(self) -> None:
        with self.assertRaises(ValueError):
            bayer_matrix(-1)

    def test_the_spot_function_screens_tile_without_a_seam(self) -> None:
        # A halftone and a line screen are built from a periodic spot function, so where the tile
        # wraps it must be no more of a step than anywhere inside it.
        for pattern in (DitherPattern.CLUSTERED, DitherPattern.LINE):
            for size in (4, 6, 8):
                tile = dither_tile(pattern, size)

                interior_height = (tile[1:] - tile[:-1]).abs().max().item()
                interior_width = (tile[:, 1:] - tile[:, :-1]).abs().max().item()

                self.assertLessEqual((tile[0] - tile[-1]).abs().max().item(), interior_height)
                self.assertLessEqual((tile[:, 0] - tile[:, -1]).abs().max().item(), interior_width)

    def test_dither_tile_rejects_too_small_a_size(self) -> None:
        with self.assertRaises(ValueError):
            dither_tile(DitherPattern.CLUSTERED, 1)

    def test_tile_to_covers_the_image_exactly(self) -> None:
        tile = dither_tile(DitherPattern.CLUSTERED, 6)
        thresholds = tile_to(tile, 37, 53, 2, (5, 3))

        self.assertEqual(thresholds.shape, (37, 53))
        self.assertTrue(bool((thresholds > 0).all() and (thresholds < 1).all()))

    def test_screens_per_element_match_tile_to(self) -> None:
        patterns = torch.tensor((0, 1, 2, 1))
        sizes = torch.tensor((4, 5, 6, 3))
        cells = torch.tensor((1, 2, 1, 1))
        transposed = torch.tensor((False, False, True, False))
        offsets = torch.tensor(((5, 3), (0, 7), (11, 2), (4, 4)))

        maps = _threshold_maps(
            patterns,
            sizes,
            cells,
            transposed,
            offsets,
            (21, 30),
            torch.device("cpu"),
            torch.float64,
        )

        for i in range(4):
            tile = dither_tile(tuple(DitherPattern)[int(patterns[i])], int(sizes[i]))
            if transposed[i]:
                tile = tile.t()

            expected = tile_to(tile, 21, 30, int(cells[i]), tuple(offsets[i].tolist()))
            self.assertTrue(torch.equal(maps[i, 0], expected))

    def test_bar_profiles_ramp_down_and_back(self) -> None:
        profiles = bar_profiles(
            8,
            torch.tensor((2, 4)),
            torch.tensor(((1.0, 0.8), (1.0, 1.0))),
            torch.tensor(((0.5, 0.6), (0.0, 0.0))),
        )

        torch.testing.assert_close(
            profiles[0], torch.tensor((1.0, 0.5, 0.5, 1.0, 0.8, 0.6, 0.6, 0.8))
        )
        torch.testing.assert_close(
            profiles[1], torch.tensor((1.0, 2 / 3, 1 / 3, 0.0, 0.0, 1 / 3, 2 / 3, 1.0))
        )

    def test_open_runs_keeps_a_line_and_drops_a_blob(self) -> None:
        images = torch.zeros(1, 1, 20, 40)
        images[0, 0, 4, 2:34] = 1.0
        images[0, 0, 12:15, 12:15] = 1.0

        opened = open_runs(images, 9, horizontal=True)

        self.assertGreater(opened[0, 0, 4, 10:26].min().item(), 0.9)
        self.assertLess(opened[0, 0, 12:15, 12:15].max().item(), 0.1)

    def test_open_runs_rejects_an_even_length(self) -> None:
        with self.assertRaises(ValueError):
            open_runs(torch.zeros(1, 1, 8, 8), 4, horizontal=True)

    def test_line_fragmentation_refuses_to_erase_a_rule(self) -> None:
        with self.assertRaises(ValueError):
            LineFragmentation(strength=(0.0, 1.0))

    def test_a_bitonal_copy_is_black_and_white_and_keeps_light_text(self) -> None:
        images = make_page(10)
        images[:, :, 30:33, 10:70] = 0.7

        output = BitonalCopy(stain_edges=(2, 2))(images, seed=0)

        # Nothing but black and white, a light gray stroke comes through black, and however
        # stained, the paper is never more than half speckle.
        self.assertTrue(bool(((output == 0) | (output == 1)).all()))
        self.assertLess(output[:, :, 31, 12:68].mean(dim=(1, 2)).max().item(), 0.1)

        speckle = 1 - output[:, :, 50:, :].mean(dim=(1, 2, 3))
        self.assertLess(speckle.max().item(), MAX_SPECKLE_DENSITY + 0.05)

    def test_a_bitonal_copy_screens_a_picture(self) -> None:
        images = make_page()
        images[:, :, 44:60, 10:40] = 0.5

        pictures = torch.zeros(1, 1, 64, 80)
        pictures[:, :, 44:60, 10:40] = 1.0

        augmentation = BitonalCopy(
            stain_edges=(0, 0), stray_density=(0.0, 0.0), pictures="pictures"
        )
        outputs = augmentation(image=Image(images), pictures=Mask(pictures), seed=0)
        output = outputs["image"]

        # The gray picture comes out as dots, part black and part white, where the same gray off a
        # picture would have gone one way or the other; the stroke off it is black.
        middle = output[0, 0, 46:58, 12:38]
        self.assertTrue(bool(((middle == 0) | (middle == 1)).all()))
        self.assertTrue(0.2 < middle.mean().item() < 0.8)
        self.assertEqual(output[0, 0, 21, 40].item(), 0.0)

        # The mask itself is left alone.
        self.assertTrue(torch.equal(outputs["pictures"], pictures))

    def test_a_bitonal_copy_needs_its_picture_mask(self) -> None:
        with self.assertRaises(ValueError):
            BitonalCopy(pictures="pictures")(make_page(), seed=0)
