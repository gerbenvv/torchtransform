"""The kinds of data that are augmented together.

Raster targets (`Image`, `Mask`) are tensors of shape `(B, C, H, W)` or `(B, C, D, H, W)` that are
resampled. Geometry targets (`Points`, `Boxes`, `RotatedBoxes`) are coordinates that are moved.
Coordinates are in pixels, in `(x, y)` or `(x, y, z)` order, with the origin in the corner of the
first pixel, so the pixel at row `i` and column `j` has its center at `(j + 0.5, i + 0.5)`.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

# A tensor per batch element, or one tensor for the whole batch.
GeometryData = torch.Tensor | Sequence[torch.Tensor]

# How a raster target is filled where the transformed canvas uncovers what lies outside the input.
Padding = str | float | tuple[float, ...]

# Padding modes that are not a fill value.
PADDING_MODES: tuple[str, ...] = ("zeros", "border", "reflection", "mean", "median")

# Interpolation modes of raster targets.
INTERPOLATION_MODES: tuple[str, ...] = ("bilinear", "nearest", "bicubic")

# Box formats.
BOX_FORMATS: tuple[str, ...] = ("xyxy", "xywh", "cxcywh")


class Target(ABC):
    """A piece of data that is augmented along with the others."""

    pass


class RasterTarget(Target):
    def __init__(
        self, data: torch.Tensor, mode: str, padding: Padding, antialias: bool, photometric: bool
    ) -> None:
        """A tensor that is resampled.

        Args:
            data: Tensor of shape `(B, C, H, W)` or `(B, C, D, H, W)`.
            mode: Interpolation, one of `"bilinear"`, `"nearest"` or `"bicubic"` (2D only).
            padding: How to fill what lies outside the input: `"zeros"`, `"border"` (repeat the
                edge), `"reflection"`, `"mean"` or `"median"` (of every channel of every element),
                or a fill value, either one number or one per channel.
            antialias: Whether to filter before sampling where a transform shrinks the data by two
                or more, so fine detail does not alias.
            photometric: Whether photometric transforms (color, noise, blur, ...) apply to it.
        """

        if not isinstance(data, torch.Tensor) or data.ndim not in (4, 5):
            raise ValueError(
                f"{type(self).__name__} needs a tensor of shape (B, C, H, W) or (B, C, D, H, W)."
            )

        if mode not in INTERPOLATION_MODES:
            raise ValueError(
                f"Unknown interpolation mode {mode!r}, expected one of {INTERPOLATION_MODES}."
            )

        if mode == "bicubic" and data.ndim == 5:
            raise ValueError("Bicubic interpolation is only available in 2D.")

        if isinstance(padding, str) and padding not in PADDING_MODES:
            raise ValueError(
                f"Unknown padding {padding!r}, expected one of {PADDING_MODES} or a value."
            )

        self.data = data
        self.mode = mode
        self.padding = padding
        self.antialias = antialias
        self.photometric = photometric

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return self.data.ndim - 2

    def to_working(self) -> torch.Tensor:
        """Returns the data as a floating point tensor to work on."""

        if self.data.is_floating_point():
            return self.data

        # Float32 holds integers exactly up to 2 ** 24 only.
        if self.data.dtype in (torch.int32, torch.int64):
            return self.data.to(dtype=torch.float64)

        return self.data.to(dtype=torch.float32)

    def from_working(self, data: torch.Tensor) -> torch.Tensor:
        """Returns worked-on data in the dtype it was given in."""

        if self.data.is_floating_point():
            return data.to(dtype=self.data.dtype)

        if self.data.dtype == torch.bool:
            return data > 0.5

        info = torch.iinfo(self.data.dtype)

        return data.round().clamp(info.min, info.max).to(dtype=self.data.dtype)


class Image(RasterTarget):
    def __init__(
        self,
        data: torch.Tensor,
        mode: str = "bilinear",
        padding: Padding = "zeros",
        antialias: bool = True,
    ) -> None:
        """An image, which both geometric and photometric transforms apply to.

        Photometric transforms expect values in `[0, 1]`. A `uint8` image is taken to be in
        `[0, 255]`: it is worked on in `[0, 1]` and given back as `uint8`.

        Args:
            data: Tensor of shape `(B, C, H, W)` or `(B, C, D, H, W)`.
            mode: Interpolation, one of `"bilinear"`, `"nearest"` or `"bicubic"` (2D only).
            padding: How to fill what lies outside the input: `"zeros"`, `"border"` (repeat the
                edge), `"reflection"`, `"mean"` or `"median"` (of every channel of every element),
                or a fill value in `[0, 1]`, either one number or one per channel.
            antialias: Whether to filter before sampling where a transform shrinks the image by two
                or more, so fine detail does not alias.
        """

        super().__init__(data, mode, padding, antialias, photometric=True)

    def to_working(self) -> torch.Tensor:
        if self.data.dtype == torch.uint8:
            return self.data.to(dtype=torch.float32) / 255

        return super().to_working()

    def from_working(self, data: torch.Tensor) -> torch.Tensor:
        if self.data.dtype == torch.uint8:
            return (data * 255).round().clamp(0, 255).to(dtype=torch.uint8)

        return super().from_working(data)


class Mask(RasterTarget):
    def __init__(
        self,
        data: torch.Tensor,
        mode: str = "nearest",
        padding: Padding = 0,
        antialias: bool = False,
    ) -> None:
        """A segmentation, or any other map that only geometric transforms apply to.

        Integer label maps keep their labels with the default nearest interpolation. Use
        `mode="bilinear"` for soft masks, heat maps or depth.

        Args:
            data: Tensor of shape `(B, C, H, W)` or `(B, C, D, H, W)`, of any dtype.
            mode: Interpolation, one of `"nearest"`, `"bilinear"` or `"bicubic"` (2D only).
            padding: How to fill what lies outside the input: a fill value (such as an ignore
                label), or `"zeros"`, `"border"`, `"reflection"`, `"mean"` or `"median"`.
            antialias: Whether to filter before sampling where a transform shrinks the map by two
                or more. Only meaningful for bilinear interpolation.
        """

        super().__init__(data, mode, padding, antialias, photometric=False)


class GeometryTarget(Target):
    """Coordinates that move with the canvas."""

    def __init__(self, data: GeometryData, values: int | None) -> None:
        """Stores the data.

        Args:
            data: A tensor of shape `(B, ..., values)`, or a sequence of `B` tensors of shape
                `(..., values)`.
            values: Number of values per item, or `None` for a number of spatial dimensions.
        """

        if isinstance(data, torch.Tensor):
            if data.ndim < 2:
                raise ValueError(f"{type(self).__name__} needs a tensor of shape (B, ..., values).")

            self.batched = True
            self.batch_size = data.shape[0]
            items = [data[i] for i in range(data.shape[0])]
        else:
            self.batched = False
            self.batch_size = len(data)
            items = list(data)

        for item in items:
            if not isinstance(item, torch.Tensor) or item.ndim < 1:
                raise ValueError(f"{type(self).__name__} needs a tensor per batch element.")

            if values is not None and item.shape[-1] != values:
                raise ValueError(f"{type(self).__name__} needs {values} values per item.")

        self.data = data
        self.items = items

        reference = items[0] if items else None
        self.device = reference.device if reference is not None else torch.device("cpu")
        self.dtype = reference.dtype if reference is not None else torch.float32

    @abstractmethod
    def encode(self, ndim: int) -> list[torch.Tensor]:
        """Returns the points to move per element, float64 CPU tensors of shape `(P_i, n)`."""

        raise NotImplementedError()

    @abstractmethod
    def decode(self, points: list[torch.Tensor], shape: tuple[int, ...]) -> GeometryData:
        """Turns the moved points back into the format the data was given in.

        Args:
            points: The moved points per batch element, of the shapes `encode` returned.
            shape: Spatial shape of the output canvas.

        Returns:
            The data in its own format, dtype and device.
        """

        raise NotImplementedError()

    def _restore(self, items: list[torch.Tensor]) -> GeometryData:
        """Returns items per batch element in the container, dtype and device they came in."""

        items = [
            x.to(
                dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
                device=self.device,
            )
            for x in items
        ]

        if self.batched:
            if items:
                return torch.stack(items)

            return self.data.clone()

        return items


class Points(GeometryTarget):
    def __init__(self, data: GeometryData) -> None:
        """Points, such as keypoints, landmarks or the vertices of polygons.

        Args:
            data: A tensor of shape `(B, ..., n)`, or a sequence of `B` tensors of shape `(..., n)`
                (such as `(N_i, n)` points or `(N_i, M_i, n)` polygons), in `(x, y[, z])` order.
        """

        super().__init__(data, values=None)

    def encode(self, ndim: int) -> list[torch.Tensor]:
        for item in self.items:
            if item.shape[-1] != ndim:
                raise ValueError(f"Points must have {ndim} coordinates to go with {ndim}D data.")

        return [
            x.detach().to(device="cpu", dtype=torch.float64).reshape(-1, ndim) for x in self.items
        ]

    def decode(self, points: list[torch.Tensor], shape: tuple[int, ...]) -> GeometryData:
        return self._restore([p.view(x.shape) for p, x in zip(points, self.items)])


class Boxes(GeometryTarget):
    def __init__(
        self, data: GeometryData, format: str = "xyxy", clip: bool = False, samples: int = 2
    ) -> None:
        """Axis-aligned bounding boxes.

        A moved box is the smallest box around its moved outline. Its corners are exact for
        affine and projective transforms; for warps that bend its edges, sample more points.

        Args:
            data: A tensor of shape `(B, N, 2 * n)`, or a sequence of `B` tensors of shape
                `(N_i, 2 * n)`.
            format: `"xyxy"` (the corners, `(x_0, y_0, x_1, y_1)`, or with `z` in 3D), `"xywh"`
                (a corner and the size) or `"cxcywh"` (the center and the size).
            clip: Whether to clip the moved boxes to the output canvas.
            samples: Points sampled along every edge, the corners included. Two samples only the
                corners.
        """

        if format not in BOX_FORMATS:
            raise ValueError(f"Unknown box format {format!r}, expected one of {BOX_FORMATS}.")

        if samples < 2:
            raise ValueError("Boxes need at least two samples per edge, the corners.")

        super().__init__(data, values=None)

        for item in self.items:
            if item.ndim != 2 or item.shape[-1] not in (4, 6):
                raise ValueError("Boxes need tensors of shape (N, 4) in 2D or (N, 6) in 3D.")

        self.format = format
        self.clip = clip
        self.samples = samples
        self._lattice: torch.Tensor | None = None

    def encode(self, ndim: int) -> list[torch.Tensor]:
        lattice = self._get_lattice(ndim)

        points = []
        for item in self.items:
            if item.shape[-1] != 2 * ndim:
                raise ValueError(f"Boxes must have {2 * ndim} values to go with {ndim}D data.")

            low, high = self._to_corners(item.detach().to(device="cpu", dtype=torch.float64), ndim)

            # Every box is its lattice of outline points, scaled between its two corners.
            outline = low[:, None] + lattice[None] * (high - low)[:, None]
            points.append(outline.reshape(-1, ndim))

        return points

    def decode(self, points: list[torch.Tensor], shape: tuple[int, ...]) -> GeometryData:
        ndim = len(shape)
        count = len(self._get_lattice(ndim))
        size = torch.tensor(shape[::-1], dtype=torch.float64)

        items = []
        for point, item in zip(points, self.items):
            outline = point.view(len(item), count, ndim)
            low, high = outline.amin(dim=1), outline.amax(dim=1)

            if self.clip:
                low = torch.minimum(low.clamp(min=0), size)
                high = torch.minimum(high.clamp(min=0), size)

            items.append(self._from_corners(low, high))

        return self._restore(items)

    def _get_lattice(self, ndim: int) -> torch.Tensor:
        """Returns the outline points of the unit box, of shape `(K, n)`."""

        if self._lattice is None or self._lattice.shape[-1] != ndim:
            steps = torch.linspace(0, 1, self.samples, dtype=torch.float64)
            lattice = torch.cartesian_prod(*(steps,) * ndim).view(-1, ndim)

            # Only the points on the outline: at least one coordinate at either end.
            on_outline = ((lattice == 0) | (lattice == 1)).any(dim=1)
            self._lattice = lattice[on_outline]

        return self._lattice

    def _to_corners(self, boxes: torch.Tensor, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the low and high corners of boxes in the configured format."""

        first, second = boxes[:, :ndim], boxes[:, ndim:]

        if self.format == "xyxy":
            return torch.minimum(first, second), torch.maximum(first, second)

        if self.format == "xywh":
            return first, first + second

        return first - 0.5 * second, first + 0.5 * second

    def _from_corners(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """Returns boxes in the configured format from their low and high corners."""

        if self.format == "xyxy":
            return torch.cat((low, high), dim=1)

        if self.format == "xywh":
            return torch.cat((low, high - low), dim=1)

        return torch.cat((0.5 * (low + high), high - low), dim=1)


class RotatedBoxes(GeometryTarget):
    def __init__(self, data: GeometryData) -> None:
        """Rotated 2D boxes.

        A moved box is the box at the new angle of its width axis that its moved corners reach
        out to. A rotation, flip or uniform scale moves it exactly.

        Args:
            data: A tensor of shape `(B, N, 5)`, or a sequence of `B` tensors of shape `(N_i, 5)`,
                in `(c_x, c_y, w, h, angle)` format. The angle is in degrees and turns the width
                axis from the x axis toward the y axis, which is clockwise on screen.
        """

        super().__init__(data, values=5)

    def encode(self, ndim: int) -> list[torch.Tensor]:
        if ndim != 2:
            raise ValueError("Rotated boxes are only available in 2D.")

        return [
            get_corners(x.detach().to(device="cpu", dtype=torch.float64).view(-1, 5)).view(-1, 2)
            for x in self.items
        ]

    def decode(self, points: list[torch.Tensor], shape: tuple[int, ...]) -> GeometryData:
        items = [fit_rotated_boxes(p.view(-1, 4, 2)) for p in points]

        return self._restore([x.view(y.shape) for x, y in zip(items, self.items)])


def get_corners(rotated_boxes: torch.Tensor) -> torch.Tensor:
    """Returns the four corners of rotated boxes.

    Args:
        rotated_boxes: Boxes of shape `(N, 5)` in `(c_x, c_y, w, h, angle)` format.

    Returns:
        Corners of shape `(N, 4, 2)`, starting at the corner that is the top-left one when the box
        is turned back upright, and going on along the width axis.
    """

    centers = rotated_boxes[:, :2]
    radians = torch.deg2rad(rotated_boxes[:, 4])
    cos, sin = torch.cos(radians), torch.sin(radians)

    # Half a step along the width axis of every box, and half a step along its height axis.
    width = 0.5 * rotated_boxes[:, 2, None] * torch.stack((cos, sin), dim=1)
    height = 0.5 * rotated_boxes[:, 3, None] * torch.stack((-sin, cos), dim=1)

    return torch.stack(
        (
            centers - width - height,
            centers + width - height,
            centers + width + height,
            centers - width + height,
        ),
        dim=1,
    )


def fit_rotated_boxes(corners: torch.Tensor) -> torch.Tensor:
    """Returns the rotated boxes that fit moved corners.

    The angle is that of the moved width axis, and the box is the smallest one at that angle that
    holds all four corners.

    Args:
        corners: Corners of shape `(N, 4, 2)`, in the order `get_corners` gives them.

    Returns:
        Boxes of shape `(N, 5)` in `(c_x, c_y, w, h, angle)` format, the angle in `(-180, 180]`.
    """

    # The width axis, as the average of the two edges along it.
    axis = (corners[:, 1] - corners[:, 0]) + (corners[:, 2] - corners[:, 3])
    radians = torch.atan2(axis[:, 1], axis[:, 0])
    cos, sin = torch.cos(radians)[:, None], torch.sin(radians)[:, None]

    # The corners in the box's own axes.
    along_width = corners[..., 0] * cos + corners[..., 1] * sin
    along_height = -corners[..., 0] * sin + corners[..., 1] * cos

    low_width, high_width = along_width.amin(dim=1), along_width.amax(dim=1)
    low_height, high_height = along_height.amin(dim=1), along_height.amax(dim=1)

    center_width = 0.5 * (low_width + high_width)
    center_height = 0.5 * (low_height + high_height)

    cos, sin = cos[:, 0], sin[:, 0]

    return torch.stack(
        (
            center_width * cos - center_height * sin,
            center_width * sin + center_height * cos,
            high_width - low_width,
            high_height - low_height,
            torch.rad2deg(radians),
        ),
        dim=1,
    )


def inside(points: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Returns which points lie on a canvas.

    Args:
        points: Points of shape `(..., n)` in pixel coordinates.
        shape: Spatial shape of the canvas, such as `(H, W)`.

    Returns:
        A boolean tensor of shape `(...)`.
    """

    size = torch.tensor(shape[::-1], dtype=points.dtype, device=points.device)

    return ((points >= 0) & (points <= size)).all(dim=-1)


def box_visibility(
    original: torch.Tensor, moved: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """Returns how much of every moved axis-aligned box still lies on the canvas.

    Use it to drop boxes (and their labels) that were moved off the canvas.

    Args:
        original: The boxes before the transform in `xyxy` format, of shape `(N, 2 * n)`, used
            for their area. Pass the moved boxes to get the share of the moved box that is visible.
        moved: The boxes after it, unclipped, of shape `(N, 2 * n)`.
        shape: Spatial shape of the output canvas.

    Returns:
        The visible volume of every moved box relative to the volume of the original box, of shape
        `(N,)`.
    """

    ndim = len(shape)
    size = torch.tensor(shape[::-1], dtype=moved.dtype, device=moved.device)

    low = torch.minimum(moved[:, :ndim].clamp(min=0), size)
    high = torch.minimum(moved[:, ndim:].clamp(min=0), size)

    visible = (high - low).clamp(min=0).prod(dim=1)
    area = (original[:, ndim:] - original[:, :ndim]).abs().prod(dim=1)

    return visible / area.clamp(min=math.ulp(1.0))
