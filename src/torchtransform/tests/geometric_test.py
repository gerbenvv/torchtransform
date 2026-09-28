from unittest import TestCase

import torch

from torchtransform.canvas import Crop, Pad, RandomCrop, RandomResizedCrop, Resize
from torchtransform.geometric import (
    Affine,
    AffineWithinBounds,
    Flip,
    HorizontalFlip,
    Perspective,
    QuarterTurn,
    RandomOrientation,
    Rotate,
    Scale,
    Shear,
    Symmetry,
    Translate,
    Transpose,
    VerticalFlip,
)
from torchtransform.targets import Boxes, Image, Mask, Points, RotatedBoxes
from torchtransform.tests.helpers import DEVICES, blobs, centroids
from torchtransform.transform import Compose, Inverse
from torchtransform.warps import ElasticWarp, GridDistortion, LensDistortion, Twirl, Wave


def transforms_2d() -> tuple:
    """Returns one of every geometric transform, set up to move content noticeably."""

    return (
        Rotate((-40, 40)),
        Scale((0.7, 1.3)),
        Scale(x=(0.8, 1.2), y=(0.8, 1.2)),
        Translate(x=(-5, 5), y=(-5, 5)),
        Translate(x=(-0.1, 0.1), relative=True),
        Shear(xy=(-15, 15), yx=(-15, 15)),
        Affine(rotation=20, scale=1.2, aspect=1.1, translation=0.05, shear=5),
        Flip(("x", "y"), p=1.0),
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
        QuarterTurn(),
        Transpose(p=1.0),
        Symmetry(),
        Perspective((0.0, 0.2)),
        # Kept to its content, since a blob pushed to the edge of the canvas is cut off.
        AffineWithinBounds(targets=()),
        ElasticWarp(magnitude=3.0, spacing=16),
        LensDistortion(0.2),
        Twirl(40, radius=(0.3, 0.6)),
        Wave(amplitude=(1, 3), wavelength=(30, 60)),
        GridDistortion(steps=4, distortion=0.15),
        Crop((40, 44)),
        RandomCrop((40, 40)),
        Pad(4),
        Resize((32, 40)),
        RandomResizedCrop((40, 40), scale=(0.5, 1.0)),
    )


def transforms_3d() -> tuple:
    """Returns geometric transforms for 3D data."""

    return (
        Rotate((-30, 30), axis="random"),
        Rotate((-30, 30), axis="x"),
        RandomOrientation(),
        Scale((0.8, 1.2), z=(0.9, 1.1)),
        Translate(x=(-3, 3), y=(-3, 3), z=(-2, 2)),
        Shear(xz=(-10, 10), zy=(-10, 10)),
        Flip(("x", "z"), p=1.0),
        QuarterTurn(axes=("y", "z")),
        Symmetry(),
        ElasticWarp(magnitude=2.0, spacing=8),
        LensDistortion(0.2),
        GridDistortion(steps=3, distortion=0.3),
        RandomCrop((16, 20, 20)),
        Resize((12, 16, 16)),
    )


class GeometricTest(TestCase):
    def test_points_follow_the_image(self) -> None:
        # A blob drawn at a point must end up where the point is moved to, for every transform.
        torch.manual_seed(0)

        shape = (48, 48)
        points = torch.rand(6, 2, dtype=torch.float64) * 16 + 16

        for transform in transforms_2d():
            for device in DEVICES:
                images = blobs(points, shape).to(device)

                outputs = transform(
                    image=Image(images, antialias=False), points=Points(points[:, None]), seed=3
                )

                moved = outputs["points"][:, 0]
                found = centroids(outputs["image"])

                name = f"{type(transform).__name__} on {device}"
                self.assertLess(float((moved - found).abs().max()), 0.6, name)

    def test_points_follow_the_image_in_3d(self) -> None:
        torch.manual_seed(0)

        shape = (24, 28, 28)
        points = torch.rand(3, 3, dtype=torch.float64) * 6 + torch.tensor((11.0, 11.0, 9.0))

        for transform in transforms_3d():
            images = blobs(points, shape)

            outputs = transform(
                image=Image(images, antialias=False), points=Points(points[:, None]), seed=5
            )

            moved = outputs["points"][:, 0]
            found = centroids(outputs["image"])

            self.assertLess(float((moved - found).abs().max()), 0.6, type(transform).__name__)

    def test_inverse_undoes_every_matrix_transform(self) -> None:
        torch.manual_seed(0)
        images = torch.rand(4, 3, 20, 24)

        for transform in transforms_2d()[:15]:
            output = Compose(transform, Inverse(transform))(images, seed=1)

            # The two fuse into the identity, so the image comes back bit for bit.
            self.assertTrue(torch.equal(output, images), type(transform).__name__)

    def test_inverse_undoes_warps_approximately(self) -> None:
        torch.manual_seed(0)
        points = torch.rand(4, 10, 2, dtype=torch.float64) * 40 + 4

        for transform in (
            ElasticWarp(magnitude=3.0),
            LensDistortion(0.2),
            Twirl(40),
            Wave(),
            GridDistortion(),
        ):
            outputs = Compose(transform, Inverse(transform))(
                points=Points(points), shape=(48, 48), seed=2
            )

            self.assertLess(
                float((outputs["points"] - points).abs().max()), 1e-6, type(transform).__name__
            )

    def test_exact_transforms_move_pixels_exactly(self) -> None:
        torch.manual_seed(0)
        images = torch.rand(8, 3, 16, 16)
        labels = torch.randint(0, 10, (8, 1, 16, 16))

        for transform in (Flip(("x", "y")), QuarterTurn(), Transpose(), Symmetry(), RandomCrop(12)):
            outputs = transform(image=Image(images), mask=Mask(labels), seed=4)

            # Every output pixel is one of the input pixels, unchanged.
            for i in range(len(images)):
                self.assertTrue(
                    bool(torch.isin(outputs["image"][i].flatten(), images[i].flatten()).all()),
                    type(transform).__name__,
                )

            self.assertEqual(outputs["mask"].dtype, torch.int64)

    def test_quarter_turns_turn_clockwise_on_screen(self) -> None:
        images = torch.zeros(1, 1, 10, 10)
        images[0, 0, 1, 7] = 1

        output = QuarterTurn(turns=1)(images)

        # The pixel at (x, y) = (7, 1) goes to (10 - 1 - 1, 7).
        self.assertEqual(output[0, 0, 7, 8].item(), 1.0)

        output = Rotate(90)(images)
        self.assertEqual(output[0, 0, 7, 8].item(), 1.0)

    def test_every_element_draws_its_own_transform(self) -> None:
        images = torch.rand(1, 3, 24, 24).expand(16, -1, -1, -1)

        for transform in (
            Rotate((-30, 30)),
            QuarterTurn(),
            Symmetry(),
            ElasticWarp(),
            RandomCrop(20),
        ):
            output = transform(images, seed=0)

            distinct = len({tuple(x.flatten()[:64].tolist()) for x in output})
            self.assertGreater(distinct, 2, type(transform).__name__)

    def test_boxes_follow_a_rotation(self) -> None:
        boxes = torch.tensor([[[10.0, 20.0, 30.0, 25.0]]])
        rotated = torch.tensor([[[20.0, 22.5, 20.0, 5.0, 0.0]]])

        outputs = Rotate(90)(
            boxes=Boxes(boxes), rotated_boxes=RotatedBoxes(rotated), shape=(40, 40), seed=0
        )

        # Turned a quarter around the center (20, 20), `(x, y)` goes to `(40 - y, x)`.
        torch.testing.assert_close(outputs["boxes"], torch.tensor([[[15.0, 10.0, 20.0, 30.0]]]))
        torch.testing.assert_close(
            outputs["rotated_boxes"], torch.tensor([[[17.5, 20.0, 20.0, 5.0, 90.0]]])
        )

    def test_affine_within_bounds_keeps_the_geometry_on_the_canvas(self) -> None:
        torch.manual_seed(0)
        points = [torch.rand(5, 2) * torch.tensor((60.0, 40.0)) for _ in range(32)]
        boxes = [torch.tensor([[1.0, 1.0, 59.0, 39.0]]) for _ in range(32)]

        transform = AffineWithinBounds(rotation=(-30, 30), scale=(0.8, 2.0))
        outputs = transform(points=Points(points), boxes=Boxes(boxes), shape=(40, 60), seed=1)

        for moved in outputs["points"] + outputs["boxes"]:
            self.assertTrue(
                bool((moved[..., 0] >= -1e-6).all() and (moved[..., 0] <= 60 + 1e-6).all())
            )
            self.assertTrue(
                bool((moved[..., 1] >= -1e-6).all() and (moved[..., 1] <= 40 + 1e-6).all())
            )

    def test_resize_shrinks_with_antialiasing(self) -> None:
        # A fine checkerboard shrunk by three comes out close to gray, rather than aliasing.
        images = ((torch.arange(63)[:, None] + torch.arange(63)[None]) % 2).float()[None, None]

        output = Resize((21, 21))(images)
        self.assertLess(float((output - 0.5).abs().max()), 0.2)

        aliased = Resize((21, 21))(Image(images, antialias=False))
        self.assertGreater(float((aliased - 0.5).abs().max()), 0.4)

    def test_resize_to_a_side(self) -> None:
        output = Resize(20)(torch.rand(2, 3, 40, 80))
        self.assertEqual(output.shape, (2, 3, 20, 40))

        output = Resize(20, side="longest")(torch.rand(2, 3, 40, 80))
        self.assertEqual(output.shape, (2, 3, 10, 20))

    def test_padding_modes(self) -> None:
        images = torch.rand(2, 3, 8, 8)

        for padding, expected in (("zeros", 0.0), (0.25, 0.25), ((0.1, 0.2, 0.3), None)):
            output = Pad(2)(Image(images, padding=padding))

            if expected is None:
                torch.testing.assert_close(
                    output[:, :, 0, 0], torch.tensor((0.1, 0.2, 0.3)).expand(2, -1)
                )
            else:
                self.assertTrue(bool((output[:, :, 0] == expected).all()))

        output = Pad(1)(Image(images, padding="border"))
        torch.testing.assert_close(output[:, :, 0, 1:-1], images[:, :, 0])

        output = Pad(1)(Image(images, padding="reflection"))
        torch.testing.assert_close(output[:, :, 0, 1:-1], images[:, :, 0])

        output = Pad(1)(Image(images, padding="mean"))
        torch.testing.assert_close(output[:, :, 0, 0], images.mean(dim=(2, 3)))

    def test_rotated_padding_fills_with_the_median(self) -> None:
        # Paper of one color with a dark square of ink on it.
        images = torch.empty(1, 3, 64, 64)
        images[:, 0], images[:, 1], images[:, 2] = 0.8, 0.9, 0.95
        images[:, :, 20:30, 20:30] = 0.0

        output = Compose(Rotate(30), Scale(0.5))(Image(images, padding="median"))

        # What the turn uncovers is the paper's color, and the ink is still there.
        torch.testing.assert_close(output[0, :, 0, 0], torch.tensor((0.8, 0.9, 0.95)))
        self.assertLess(float(output.min()), 0.1)

    def test_elastic_warp_before_a_zoom_out_stays_in_bounds(self) -> None:
        # Coordinates beyond the canvas reach the end of the spline, in any precision.
        for shape, dtype in (((1024, 1024), torch.float32), ((64, 2048), torch.float64)):
            images = torch.rand(1, 1, *shape, dtype=dtype)

            for transform in (Scale(0.8), Rotate(20)):
                output = Compose(ElasticWarp(), transform)(images, seed=0)
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_turns_are_exact_on_canvases_of_mixed_parity(self) -> None:
        images = torch.arange(20.0).view(1, 1, 5, 4)

        for transform in (Transpose(p=1.0), QuarterTurn(turns=1), Symmetry()):
            output = transform(images, seed=0)

            # Every pixel is one of the input pixels, not a blend of two.
            self.assertTrue(bool(torch.isin(output, images).all()), type(transform).__name__)

    def test_lens_distortion_keeps_geometry_finite(self) -> None:
        boxes = torch.tensor([[[0.0, 0.0, 64.0, 64.0]]])

        output = LensDistortion((-0.2, -0.2))(boxes=Boxes(boxes), shape=(64, 64))["boxes"]
        self.assertTrue(bool(torch.isfinite(output).all()))

        output = Compose(LensDistortion((-0.2, -0.1)), AffineWithinBounds())(
            torch.rand(2, 3, 32, 32)
        )
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_inverse_after_a_change_of_canvas(self) -> None:
        points = torch.tensor([[[10.0, 10.0]]])
        move = Translate(x=0.25, relative=True)

        outputs = Compose(move, Crop(32), Inverse(move))(points=Points(points), shape=(64, 64))

        # Moved by a quarter of 64, cropped by 16, and moved back by the same 16.
        torch.testing.assert_close(outputs["points"], torch.tensor([[[-6.0, -6.0]]]))

    def test_inverse_of_a_canvas_transform_keeps_to_its_elements(self) -> None:
        points = torch.full((64, 1, 2), 32.0)
        crop = RandomCrop(32)

        outputs = Compose(crop, Inverse(crop, p=0.5))(points=Points(points), shape=(64, 64), seed=1)
        cropped = crop(points=Points(points), shape=(64, 64), seed=1)["points"]

        # Inverted elements are back where they were, the others centered on the new canvas.
        back = (outputs["points"] == points).all(dim=2)[:, 0]
        centered = (outputs["points"] == cropped + 16).all(dim=2)[:, 0]

        self.assertTrue(bool((back | centered).all()))
        self.assertTrue(10 < int(back.sum()) < 54)
