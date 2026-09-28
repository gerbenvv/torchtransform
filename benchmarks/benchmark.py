"""Benchmarks torchtransform against kornia and torchvision on the same pipelines.

Every library augments every batch element with its own random parameters: torchtransform and
kornia batched, torchvision one sample at a time (its batched transforms draw one set of
parameters for the whole batch). "unfused" is torchtransform with a no-op pixel transform after
every transform, which forces a resampling after each, as a sequential library would do.

Run with `python benchmarks/benchmark.py`. It needs `kornia` and `torchvision` next to
torchtransform, and prints a Markdown table.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import torch

import torchtransform as tt

# A function that augments a batch of images and masks.
Pipeline = Callable[[torch.Tensor, torch.Tensor], object]


def unfused(*transforms: tt.Transform) -> tt.Compose:
    """Returns a composition that resolves after every transform, as if nothing were fused."""

    steps: list[tt.Transform] = []
    for transform in transforms:
        steps += [transform, tt.Lambda(lambda x: x)]

    return tt.Compose(*steps)


def classification_pipelines(size: int) -> dict[str, Pipeline]:
    """Returns the classic classification pipeline in every library."""

    transforms = (
        tt.RandomResizedCrop(size),
        tt.HorizontalFlip(),
        tt.ColorJitter(0.4, 0.4, 0.4, 0.1),
    )
    fused = tt.Compose(*transforms)
    separate = unfused(*transforms)

    pipelines: dict[str, Pipeline] = {
        "torchtransform": lambda x, m: fused(x),
        "torchtransform, unfused": lambda x, m: separate(x),
    }

    try:
        import kornia.augmentation as K

        kornia = K.AugmentationSequential(
            K.RandomResizedCrop((size, size)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0),
        )
        pipelines["kornia"] = lambda x, m: kornia(x)
    except ImportError:
        pass

    try:
        from torchvision.transforms import v2

        torchvision = v2.Compose(
            [
                v2.RandomResizedCrop(size, antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ]
        )
        pipelines["torchvision, per sample"] = lambda x, m: [torchvision(y) for y in x]
    except ImportError:
        pass

    return pipelines


def geometric_pipelines(size: int) -> dict[str, Pipeline]:
    """Returns a heavy geometric pipeline, on images and masks, in every library."""

    transforms = (
        tt.Affine(rotation=30, scale=1.2, translation=0.1, shear=10),
        tt.Perspective((0.0, 0.2)),
        tt.ElasticWarp(magnitude=(0.0, 8.0), spacing=32),
        tt.Crop(size),
    )
    fused = tt.Compose(*transforms)
    separate = unfused(*transforms)

    pipelines: dict[str, Pipeline] = {
        "torchtransform": lambda x, m: fused(image=x, mask=tt.Mask(m)),
        "torchtransform, unfused": lambda x, m: separate(image=x, mask=tt.Mask(m)),
    }

    try:
        import kornia.augmentation as K

        kornia = K.AugmentationSequential(
            K.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(1 / 1.2, 1.2), shear=10, p=1.0),
            K.RandomPerspective(0.2, p=1.0),
            K.RandomElasticTransform(alpha=(8.0, 8.0), p=1.0),
            K.CenterCrop(size),
            data_keys=["input", "mask"],
        )
        pipelines["kornia"] = lambda x, m: kornia(x, m.float())
    except ImportError:
        pass

    try:
        from torchvision import tv_tensors
        from torchvision.transforms import v2

        torchvision = v2.Compose(
            [
                v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(1 / 1.2, 1.2), shear=10),
                v2.RandomPerspective(0.2, p=1.0),
                v2.ElasticTransform(alpha=50.0),
                v2.CenterCrop(size),
            ]
        )
        pipelines["torchvision, per sample"] = lambda x, m: [
            torchvision(y, tv_tensors.Mask(n)) for y, n in zip(x, m)
        ]
    except ImportError:
        pass

    return pipelines


def volume_pipelines(size: int) -> dict[str, Pipeline]:
    """Returns a 3D pipeline, on volumes and masks."""

    transforms = (
        tt.Rotate((-30, 30), axis="random"),
        tt.Scale((0.9, 1.1)),
        tt.Flip(("x", "y", "z"), p=0.5),
        tt.ElasticWarp(magnitude=(0.0, 4.0), spacing=24),
        tt.Crop(size),
    )
    fused = tt.Compose(*transforms)
    separate = unfused(*transforms)

    return {
        "torchtransform": lambda x, m: fused(image=x, mask=tt.Mask(m)),
        "torchtransform, unfused": lambda x, m: separate(image=x, mask=tt.Mask(m)),
    }


def measure(pipeline: Pipeline, images: torch.Tensor, masks: torch.Tensor, repeats: int) -> float:
    """Returns the median time of a pipeline in milliseconds."""

    def synchronize() -> None:
        if images.is_cuda:
            torch.cuda.synchronize()

    for _ in range(3):
        pipeline(images, masks)

    times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()

        pipeline(images, masks)

        synchronize()
        times.append(1000 * (time.perf_counter() - start))

    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--devices", nargs="+", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()

    torch.manual_seed(0)

    scenarios = (
        (
            "Classification: random resized crop, flip, color jitter",
            classification_pipelines(224),
            (64, 3, 512, 512),
        ),
        (
            "Geometric: affine, perspective, elastic, crop, with masks",
            geometric_pipelines(384),
            (32, 3, 512, 512),
        ),
        (
            "3D: rotation, scale, flips, elastic, crop, with masks",
            volume_pipelines(64),
            (4, 1, 96, 96, 96),
        ),
    )

    print("| Pipeline | Library | " + " | ".join(arguments.devices) + " |")
    print("| --- | --- | " + " | ".join("---:" for _ in arguments.devices) + " |")

    with torch.no_grad():
        for title, pipelines, shape in scenarios:
            data = {
                device: (
                    torch.rand(shape, device=device),
                    torch.randint(0, 4, (shape[0], 1, *shape[2:]), device=device),
                )
                for device in arguments.devices
            }

            for name, pipeline in pipelines.items():
                cells = []
                for device in arguments.devices:
                    images, masks = data[device]
                    repeats = (
                        arguments.repeats if device != "cpu" else max(3, arguments.repeats // 3)
                    )
                    cells.append(f"{measure(pipeline, images, masks, repeats):.1f} ms")

                print(f"| {title} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
