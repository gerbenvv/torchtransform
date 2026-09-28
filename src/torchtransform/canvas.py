"""Transforms that change the shape of the canvas: crops, pads and resizes.

They are fused into a single resampling with the other geometric transforms, so a crop at the end
of a pipeline means only the cropped pixels are ever computed. Crops and pads by whole pixels,
alone or with flips and quarter turns, are exact: pixels are moved, not interpolated.
"""

import torch

from torchtransform import matrices as mat
from torchtransform.distributions import Parameter, to_distribution
from torchtransform.state import Context
from torchtransform.transform import CanvasTransform, centered_placement

# A spatial shape: one size for every axis, or a size per axis (in tensor order, such as `(H, W)`),
# where `None` keeps that axis.
ShapeParameter = int | tuple[int | None, ...]

# Ways to resize to a shape of another aspect ratio.
FITS: tuple[str, ...] = ("stretch", "contain", "cover")


def resolve_shape(shape: ShapeParameter, current: tuple[int, ...]) -> tuple[int, ...]:
    """Returns a spatial shape for a canvas, from a shape parameter."""

    if isinstance(shape, int):
        return (shape,) * len(current)

    if len(shape) != len(current):
        raise ValueError(
            f"The shape {shape} does not have one size for each of the axes of {current}."
        )

    return tuple(c if s is None else int(s) for s, c in zip(shape, current))


def size_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """Returns a spatial shape as a size in `(x, y[, z])` order, as a float64 tensor."""

    return torch.tensor(shape[::-1], dtype=torch.float64)


class Crop(CanvasTransform):
    def __init__(self, shape: ShapeParameter | None = None) -> None:
        """Crops to a shape around the center, and pads where the canvas is smaller.

        Args:
            shape: The spatial shape, such as `(H, W)`, or one size for every axis. By default the
                shape of the input, which undoes the canvas changes before it.
        """

        super().__init__()

        self.shape = shape

    def get_output_shape(self, context: Context) -> tuple[int, ...]:
        if self.shape is None:
            return context.state.input_shape

        return resolve_shape(self.shape, context.shape)

    def get_matrices(self, context: Context, output_shape: tuple[int, ...]) -> torch.Tensor:
        return centered_placement(context.batch_size, context.shape, output_shape)

    def extra_repr(self) -> str:
        return f"shape={self.shape}"


class CenterCrop(Crop):
    """The same as `Crop`, by the name other libraries use."""

    pass


class RandomCrop(CanvasTransform):
    def __init__(self, shape: ShapeParameter) -> None:
        """Crops to a shape at a random position on whole pixels, drawn per element.

        Where the canvas is smaller than the shape, it is padded, and placed at a random position
        within the padding instead.

        Args:
            shape: The spatial shape, such as `(H, W)`, or one size for every axis.
        """

        super().__init__()

        self.shape = shape

    def get_output_shape(self, context: Context) -> tuple[int, ...]:
        return resolve_shape(self.shape, context.shape)

    def get_matrices(self, context: Context, output_shape: tuple[int, ...]) -> torch.Tensor:
        input_size, output_size = size_tensor(context.shape), size_tensor(output_shape)

        # The offset of the crop in the input, on whole pixels, of either sign.
        difference = input_size - output_size
        fractions = context.rand(context.batch_size, context.ndim)
        offsets = torch.floor(fractions * (difference.abs() + 1)) * torch.sign(difference)

        return mat.translation(0.5 * difference - offsets)

    def extra_repr(self) -> str:
        return f"shape={self.shape}"


class Pad(CanvasTransform):
    def __init__(self, padding: int | tuple[int, ...]) -> None:
        """Pads the canvas.

        Args:
            padding: Pixels to add: one number for every side, one per axis (in tensor order, such
                as `(H, W)`) for both of its sides, or one per side, first the start of every axis
                and then the end, in tensor order: `(top, left, bottom, right)` in 2D.
        """

        super().__init__()

        self.padding = padding

    def _get_padding(self, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the padding at the start and at the end of every axis, in coordinate order."""

        padding = self.padding
        if isinstance(padding, int):
            padding = (padding,) * ndim

        if len(padding) == ndim:
            padding = tuple(padding) * 2

        if len(padding) != 2 * ndim:
            raise ValueError(f"Padding {self.padding} does not fit {ndim}D data.")

        before = torch.tensor(padding[:ndim][::-1], dtype=torch.float64)
        after = torch.tensor(padding[ndim:][::-1], dtype=torch.float64)

        return before, after

    def get_output_shape(self, context: Context) -> tuple[int, ...]:
        before, after = self._get_padding(context.ndim)
        added = (before + after).flip(0).to(dtype=torch.int64).tolist()

        return tuple(s + a for s, a in zip(context.shape, added))

    def get_matrices(self, context: Context, output_shape: tuple[int, ...]) -> torch.Tensor:
        before, _ = self._get_padding(context.ndim)
        shift = 0.5 * size_tensor(context.shape) + before - 0.5 * size_tensor(output_shape)

        return mat.translation(shift.expand(context.batch_size, -1))

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


class Resize(CanvasTransform):
    def __init__(self, shape: ShapeParameter, fit: str = "stretch", side: str = "shortest") -> None:
        """Resizes to a shape.

        Shrinking by two or more is antialiased for images (see `Image`).

        Args:
            shape: The spatial shape, such as `(H, W)`. A single number is the size of one side
                instead, with the aspect ratio kept.
            fit: For a shape of another aspect ratio: `"stretch"` to it, `"contain"` the content
                within it (padding the rest), or `"cover"` it with the content (cropping the rest).
            side: For a single number, the side it is the size of: `"shortest"` or `"longest"`.
        """

        super().__init__()

        if fit not in FITS:
            raise ValueError(f"Unknown fit {fit!r}, expected one of {FITS}.")

        if side not in ("shortest", "longest"):
            raise ValueError(f"Unknown side {side!r}, expected 'shortest' or 'longest'.")

        self.shape = shape
        self.fit = fit
        self.side = side

    def get_output_shape(self, context: Context) -> tuple[int, ...]:
        if isinstance(self.shape, int):
            factor = self.shape / (
                min(context.shape) if self.side == "shortest" else max(context.shape)
            )

            return tuple(max(1, round(s * factor)) for s in context.shape)

        return resolve_shape(self.shape, context.shape)

    def get_matrices(self, context: Context, output_shape: tuple[int, ...]) -> torch.Tensor:
        factors = size_tensor(output_shape) / size_tensor(context.shape)

        if self.fit == "contain" and not isinstance(self.shape, int):
            factors = factors.min().expand(context.ndim)
        elif self.fit == "cover" and not isinstance(self.shape, int):
            factors = factors.max().expand(context.ndim)

        return mat.scale(factors.expand(context.batch_size, -1))

    def extra_repr(self) -> str:
        return f"shape={self.shape}, fit={self.fit!r}"


class RandomResizedCrop(CanvasTransform):
    def __init__(
        self,
        shape: ShapeParameter,
        scale: Parameter = (0.08, 1.0),
        ratio: Parameter = (3 / 4, 4 / 3),
    ) -> None:
        """Crops a random part and resizes it to a shape, drawn per element.

        The crop covers a random fraction of the canvas, at a random aspect ratio and position.
        Shrinking by two or more is antialiased for images (see `Image`).

        Args:
            shape: The spatial shape, such as `(H, W)`, or one size for every axis.
            scale: Fraction of the area (or volume) of the canvas the crop covers.
            ratio: Aspect ratio of the crop, width over height, drawn uniformly in the logarithm,
                relative to that of the shape.
        """

        super().__init__()

        self.shape = shape
        self.scale = to_distribution(scale)
        self.ratio = to_distribution(ratio, log=True)

    def get_output_shape(self, context: Context) -> tuple[int, ...]:
        return resolve_shape(self.shape, context.shape)

    def get_matrices(self, context: Context, output_shape: tuple[int, ...]) -> torch.Tensor:
        batch_size, ndim = context.batch_size, context.ndim
        input_size, output_size = size_tensor(context.shape), size_tensor(output_shape)

        # The crop has the shape of the output, scaled to cover the drawn fraction of the input,
        # and stretched by the drawn ratio.
        fractions = context.sample(self.scale)
        stretches = torch.ones(batch_size, ndim, dtype=torch.float64)
        stretches[:, 0] = torch.sqrt(context.sample(self.ratio))
        stretches[:, 1] = 1 / stretches[:, 0]

        volume = float(input_size.prod())
        base = output_size * (volume / float(output_size.prod())) ** (1 / ndim)
        sizes = base * stretches * (fractions ** (1 / ndim))[:, None]
        sizes = torch.minimum(sizes, input_size)

        # Its center, where it still lies within the input.
        room = 0.5 * (input_size - sizes)
        centers = (2 * context.rand(batch_size, ndim) - 1) * room

        factors = output_size / sizes

        return mat.scale(factors) @ mat.translation(-centers)

    def extra_repr(self) -> str:
        return f"shape={self.shape}, scale={self.scale}, ratio={self.ratio}"
