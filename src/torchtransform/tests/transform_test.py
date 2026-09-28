from unittest import TestCase

import torch

from torchtransform.canvas import Crop, RandomCrop, RandomResizedCrop, Resize
from torchtransform.color import Brightness, Invert
from torchtransform.geometric import HorizontalFlip, QuarterTurn, Rotate, Scale, Translate
from torchtransform.targets import Boxes, Image, Mask, Points
from torchtransform.transform import (
    Compose,
    Identity,
    Inverse,
    Lambda,
    Maybe,
    OneOf,
    PixelTransform,
    SomeOf,
)
from torchtransform.warps import ElasticWarp


class AddOne(PixelTransform):
    """Adds one, to count how often a transform is applied."""

    def apply_image(self, image, parameters, context):
        return image + 1


class TransformTest(TestCase):
    def test_a_plain_tensor_is_an_image(self) -> None:
        images = torch.rand(2, 3, 8, 8)

        output = Invert()(images)
        torch.testing.assert_close(output, 1 - images)

        first, second = Invert()(images, images)
        torch.testing.assert_close(first, second)

        outputs = Invert()(a=images, b=Mask(images))
        torch.testing.assert_close(outputs["a"], 1 - images)
        torch.testing.assert_close(outputs["b"], images)

    def test_inputs_are_never_changed(self) -> None:
        images = torch.rand(2, 3, 8, 8)
        before = images.clone()

        Compose(AddOne(), Invert(), HorizontalFlip(p=1.0), AddOne())(images)

        self.assertTrue(torch.equal(images, before))

    def test_probability_is_drawn_per_element(self) -> None:
        images = torch.zeros(1000, 1, 1, 1)

        output = AddOne(p=0.3)(images, seed=0)
        self.assertTrue(0.25 < float(output.mean()) < 0.35)

        output = Maybe(AddOne(), p=0.7)(images, seed=0)
        self.assertTrue(0.65 < float(output.mean()) < 0.75)

    def test_one_of_applies_exactly_one(self) -> None:
        images = torch.zeros(500, 1, 1, 1)

        output = OneOf(AddOne(), AddOne(), AddOne())(images, seed=0)
        self.assertTrue(bool((output == 1).all()))

        output = OneOf(AddOne(), Identity(), weights=(3, 1))(images, seed=0)
        self.assertTrue(0.7 < float(output.mean()) < 0.8)

    def test_some_of_applies_a_number_in_its_range(self) -> None:
        images = torch.zeros(500, 1, 1, 1)

        output = SomeOf(AddOne(), AddOne(), AddOne(), n=2)(images, seed=0)
        self.assertTrue(bool((output == 2).all()))

        output = SomeOf(AddOne(), AddOne(), AddOne(), AddOne(), n=(1, 3))(images, seed=0)
        self.assertEqual(set(output.flatten().tolist()), {1.0, 2.0, 3.0})

        output = SomeOf(AddOne(), AddOne(), n=0)(images, seed=0)
        self.assertTrue(bool((output == 0).all()))

    def test_masked_elements_are_untouched(self) -> None:
        torch.manual_seed(0)
        images = torch.rand(64, 3, 16, 16)

        for transform in (
            Rotate((-30, 30), p=0.5),
            ElasticWarp(p=0.5),
            Brightness(p=0.5),
            AddOne(p=0.5),
        ):
            output = transform(images, seed=1)

            unchanged = (output == images).flatten(1).all(dim=1)
            self.assertTrue(10 < int(unchanged.sum()) < 54, type(transform).__name__)

    def test_calls_are_deterministic_under_a_seed(self) -> None:
        images = torch.rand(8, 3, 16, 16)
        transform = Compose(Rotate((-30, 30)), ElasticWarp(), Brightness(p=0.5), RandomCrop(12))

        first = transform(images, seed=5)
        second = transform(images, seed=5)
        third = transform(images, seed=6)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

        # Without a seed, torch's global generator decides.
        torch.manual_seed(3)
        first = transform(images)
        torch.manual_seed(3)
        second = transform(images)

        self.assertTrue(torch.equal(first, second))

    def test_draws_do_not_depend_on_the_rest_of_the_pipeline(self) -> None:
        images = torch.rand(8, 3, 16, 16)
        rotation = Rotate((-30, 30)).manual_seed(42)

        alone = rotation(images, seed=1)
        after = Compose(Translate(x=0.0), Maybe(Brightness(), p=0.0), rotation)(images, seed=1)

        self.assertTrue(torch.equal(alone, after))

    def test_same_seed_gives_the_same_draws(self) -> None:
        images = torch.rand(8, 3, 16, 16)

        first = Rotate((-30, 30)).manual_seed(7)(images, seed=1)
        second = Rotate((-30, 30)).manual_seed(7)(images, seed=1)

        self.assertTrue(torch.equal(first, second))

    def test_inverse_of_a_composition(self) -> None:
        images = torch.rand(4, 3, 16, 16)
        inner = Compose(Rotate((-30, 30)), Scale((0.8, 1.2)), Maybe(QuarterTurn(), p=0.5))

        output = Compose(inner, Inverse(inner))(images, seed=3)
        torch.testing.assert_close(output, images)

    def test_intermediate_canvases_do_not_cut_content_off(self) -> None:
        images = torch.rand(4, 3, 16, 20)
        crop = RandomCrop((10, 10))

        # The chain is one map from the output canvas to the input, so a crop and its inverse
        # give back the whole image, and not a padded crop.
        output = Compose(crop, Inverse(crop))(images)
        self.assertTrue(torch.equal(output, images))

        with self.assertRaises(RuntimeError):
            Inverse(crop)(images)

    def test_replay_inverse_fills_what_was_cropped(self) -> None:
        images = torch.rand(4, 3, 16, 20)

        output, replay = RandomCrop((10, 10))(images, replay=True)
        restored = replay.inverse(Image(output, padding=-1.0))

        self.assertEqual(restored.shape, images.shape)

        kept = restored != -1
        self.assertEqual(int(kept.sum()), 4 * 3 * 10 * 10)
        self.assertTrue(torch.equal(restored[kept], images[kept]))

    def test_replay_applies_and_undoes_the_geometry(self) -> None:
        images = torch.rand(4, 3, 32, 32)
        masks = torch.randint(0, 3, (4, 1, 32, 32))
        points = torch.rand(4, 5, 2) * 32

        transform = Compose(
            Rotate((-20, 20)), Brightness(), RandomResizedCrop(24, scale=(0.5, 1.0))
        )
        output, replay = transform(images, seed=2, replay=True)

        # The same geometry applied to the masks afterward.
        together = transform(image=images, mask=Mask(masks), seed=2)
        self.assertTrue(torch.equal(replay.apply(Mask(masks)), together["mask"]))

        # Undone, the points come back where they were.
        moved = replay.apply(points=Points(points))["points"]
        restored = replay.inverse(points=Points(moved))["points"]
        torch.testing.assert_close(restored, points)

        self.assertEqual(replay.inverse(output).shape, images.shape)

    def test_canvas_transforms_apply_to_the_whole_batch(self) -> None:
        images = torch.rand(8, 3, 16, 16)

        output = Maybe(Resize((8, 8)), p=0.5)(images, seed=1)
        self.assertEqual(output.shape, (8, 3, 8, 8))

        # Elements it does not apply to are left unscaled in the center of the new canvas.
        output = OneOf(Resize((8, 8)), Crop((8, 8)))(images, seed=1)
        self.assertEqual(output.shape, (8, 3, 8, 8))

    def test_geometry_only(self) -> None:
        boxes = [torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.zeros(0, 4)]

        outputs = HorizontalFlip(p=1.0)(boxes=Boxes(boxes), shape=(10, 10))

        torch.testing.assert_close(outputs["boxes"][0], torch.tensor([[7.0, 2.0, 9.0, 4.0]]))
        self.assertEqual(outputs["boxes"][1].shape, (0, 4))

    def test_lambda(self) -> None:
        images = torch.rand(4, 3, 8, 8)

        output = Lambda(lambda x: x * 0)(images)
        self.assertTrue(bool((output == 0).all()))

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            Rotate(10)(torch.rand(2, 3, 8, 8), torch.rand(3, 3, 8, 8))

        with self.assertRaises(ValueError):
            Rotate(10)(torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 9))

        with self.assertRaises(ValueError):
            Rotate(10)(torch.rand(2, 3, 8, 8), b=torch.rand(2, 3, 8, 8))

        with self.assertRaises(ValueError):
            Rotate(10, p=1.5)

        with self.assertRaises(NotImplementedError):
            Inverse(AddOne())(torch.rand(2, 3, 8, 8))

    def test_modules_print(self) -> None:
        transform = Compose(Rotate((-10, 10)), Maybe(Brightness(), p=0.5))

        self.assertIn("Rotate", repr(transform))
        self.assertIn("p=0.5", repr(transform))
