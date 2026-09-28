from unittest import TestCase

import torch

from torchtransform.geometric import HorizontalFlip, Rotate, Scale, Translate
from torchtransform.targets import (
    Boxes,
    Image,
    Mask,
    Points,
    RotatedBoxes,
    box_visibility,
    fit_rotated_boxes,
    get_corners,
    inside,
)
from torchtransform.transform import Compose
from torchtransform.warps import ElasticWarp


class TargetsTest(TestCase):
    def test_uint8_images_are_worked_on_in_zero_to_one(self) -> None:
        images = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)

        output = HorizontalFlip(p=1.0)(images)

        self.assertEqual(output.dtype, torch.uint8)
        self.assertTrue(torch.equal(output, images.flip(-1)))

    def test_integer_masks_keep_their_labels(self) -> None:
        labels = torch.randint(0, 7, (4, 1, 16, 16))

        output = Rotate((-30, 30))(Mask(labels, padding=255))

        self.assertEqual(output.dtype, labels.dtype)
        self.assertTrue(set(output.unique().tolist()) <= set(range(7)) | {255})

    def test_points_keep_their_format(self) -> None:
        points = [
            torch.rand(3, 4, 2),
            torch.rand(0, 4, 2),
            torch.rand(1, 4, 2, dtype=torch.float64),
        ]

        outputs = HorizontalFlip(p=1.0)(points=Points(points), shape=(10, 10))["points"]

        self.assertIsInstance(outputs, list)
        self.assertEqual([x.shape for x in outputs], [x.shape for x in points])
        torch.testing.assert_close(outputs[0][..., 0], 10 - points[0][..., 0])

        batched = torch.rand(2, 5, 2)
        outputs = HorizontalFlip(p=1.0)(points=Points(batched), shape=(10, 10))["points"]
        self.assertEqual(outputs.shape, batched.shape)

    def test_box_formats(self) -> None:
        xyxy = torch.tensor([[[1.0, 2.0, 5.0, 8.0]]])
        xywh = torch.tensor([[[1.0, 2.0, 4.0, 6.0]]])
        cxcywh = torch.tensor([[[3.0, 5.0, 4.0, 6.0]]])

        outputs = HorizontalFlip(p=1.0)(
            a=Boxes(xyxy),
            b=Boxes(xywh, format="xywh"),
            c=Boxes(cxcywh, format="cxcywh"),
            shape=(10, 10),
        )

        torch.testing.assert_close(outputs["a"], torch.tensor([[[5.0, 2.0, 9.0, 8.0]]]))
        torch.testing.assert_close(outputs["b"], torch.tensor([[[5.0, 2.0, 4.0, 6.0]]]))
        torch.testing.assert_close(outputs["c"], torch.tensor([[[7.0, 5.0, 4.0, 6.0]]]))

    def test_boxes_are_clipped(self) -> None:
        boxes = torch.tensor([[[2.0, 2.0, 8.0, 8.0]]])

        outputs = Scale(2.0)(boxes=Boxes(boxes, clip=True), shape=(10, 10))
        torch.testing.assert_close(outputs["boxes"], torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]))

    def test_boxes_in_3d(self) -> None:
        boxes = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])

        outputs = HorizontalFlip(p=1.0)(boxes=Boxes(boxes), shape=(8, 10, 10))
        torch.testing.assert_close(
            outputs["boxes"], torch.tensor([[[6.0, 2.0, 3.0, 9.0, 5.0, 6.0]]])
        )

    def test_sampled_box_edges_follow_warps(self) -> None:
        boxes = torch.tensor([[[10.0, 10.0, 40.0, 40.0]]])
        warp = ElasticWarp(magnitude=4.0, spacing=8)

        corners = warp(boxes=Boxes(boxes), shape=(50, 50), seed=0)["boxes"]
        sampled = warp(boxes=Boxes(boxes, samples=9), shape=(50, 50), seed=0)["boxes"]

        # More samples along the edges can only grow the box.
        self.assertTrue(bool((sampled[..., :2] <= corners[..., :2] + 1e-9).all()))
        self.assertTrue(bool((sampled[..., 2:] >= corners[..., 2:] - 1e-9).all()))

    def test_rotated_boxes_round_trip_through_their_corners(self) -> None:
        boxes = torch.tensor(
            [[10.0, 20.0, 8.0, 4.0, 30.0], [5.0, 5.0, 2.0, 6.0, -120.0]], dtype=torch.float64
        )

        torch.testing.assert_close(fit_rotated_boxes(get_corners(boxes)), boxes)

    def test_rotated_boxes_follow_rotations_and_flips(self) -> None:
        boxes = torch.tensor([[[20.0, 10.0, 8.0, 4.0, 10.0]]])

        outputs = Compose(Rotate(25), HorizontalFlip(p=1.0))(
            rotated=RotatedBoxes(boxes), shape=(40, 40)
        )
        moved = outputs["rotated"][0, 0]

        # The size stays, and the corners land where the moved corners of the original do.
        torch.testing.assert_close(moved[2:4], boxes[0, 0, 2:4])

        corners = Compose(Rotate(25), HorizontalFlip(p=1.0))(
            points=Points(get_corners(boxes[0])[None]), shape=(40, 40)
        )["points"][0, 0]

        found = get_corners(moved[None].double())[0].float()
        distances = torch.cdist(found, corners.float())
        self.assertLess(float(distances.min(dim=1).values.max()), 1e-4)

    def test_rotated_boxes_need_2d(self) -> None:
        with self.assertRaises(ValueError):
            Rotate(10)(boxes=RotatedBoxes(torch.zeros(1, 1, 5)), shape=(4, 4, 4))

    def test_inside_and_visibility(self) -> None:
        points = torch.tensor([[1.0, 1.0], [11.0, 1.0], [-1.0, 5.0]])
        self.assertEqual(inside(points, (10, 10)).tolist(), [True, False, False])

        original = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        moved = torch.tensor([[8.0, 8.0, 12.0, 12.0]])
        torch.testing.assert_close(box_visibility(original, moved, (10, 10)), torch.tensor([0.25]))

    def test_invalid_targets(self) -> None:
        with self.assertRaises(ValueError):
            Image(torch.rand(3, 8, 8))

        with self.assertRaises(ValueError):
            Image(torch.rand(1, 3, 8, 8), padding="wrap")

        with self.assertRaises(ValueError):
            Image(torch.rand(1, 3, 4, 8, 8), mode="bicubic")

        with self.assertRaises(ValueError):
            Boxes(torch.rand(1, 2, 5))

    def test_boolean_masks(self) -> None:
        masks = torch.rand(2, 1, 8, 8) > 0.5

        output = Rotate(10)(Mask(masks))
        self.assertEqual(output.dtype, torch.bool)

    def test_large_labels_stay_exact(self) -> None:
        labels = torch.full((1, 1, 4, 4), 2**24 + 1)

        self.assertTrue(torch.equal(HorizontalFlip(p=1.0)(Mask(labels)), labels))
        self.assertTrue(torch.equal(Rotate(90)(Mask(labels)), labels))

    def test_nearest_padding_never_mixes_label_and_fill(self) -> None:
        labels = torch.full((1, 1, 4, 4), 3)

        for distance in (0.5, -0.5, 1.5, 0.4999, 0.5001):
            output = Translate(x=distance)(Mask(labels, padding=255))
            self.assertTrue(set(output.unique().tolist()) <= {3, 255}, distance)

    def test_bicubic_padding_fills_exactly(self) -> None:
        images = torch.full((1, 3, 32, 32), 0.5)

        for padding in (0.5, "mean"):
            output = Rotate(30)(Image(images, mode="bicubic", padding=padding))
            torch.testing.assert_close(output, images)
