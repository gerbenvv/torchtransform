"""Resolving a chain of coordinate transforms into a single resampling.

Every geometric transform adds a step to a chain: a matrix per batch element, or a warp. The chain
is only resolved when pixels are needed, and then in one go. Consecutive matrices are multiplied
into one, crops and pads are part of them, and the whole chain becomes one of:

- An index gather, when every element is only flipped, turned by quarter turns, transposed and
  moved by whole pixels. This is exact and involves no interpolation at all.
- One `affine_grid` and `grid_sample`, for any affine chain, or one projective grid.
- A warp grid, evaluated at a coarse resolution (warps are smooth) and upsampled, followed by one
  `grid_sample`.

All steps work in centered coordinates: pixels, with the origin in the center of the canvas, so
rotations and scales are around the center and a centered crop or pad is the identity.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from torchtransform import matrices as mat
from torchtransform.targets import RasterTarget

# Warps are called with coordinates of shape `(B, ..., n)` and their parameters.
WarpFunction = Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]


@dataclass
class MatrixStep:
    """A step of matrices, from the canvas before it to the canvas after it."""

    matrices: torch.Tensor  # Forward matrices of shape `(B, n + 1, n + 1)`, float64 on the CPU.
    input_shape: tuple[int, ...]  # Spatial shape of the canvas before the step.
    output_shape: tuple[int, ...]  # Spatial shape of the canvas after the step.

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Moves coordinates of shape `(B, ..., n)` from the input canvas to the output canvas."""

        return mat.apply(self.matrices, coordinates)

    def inverted(self) -> "MatrixStep":
        """Returns the step that undoes this one."""

        return MatrixStep(torch.linalg.inv(self.matrices), self.output_shape, self.input_shape)

    def subset(self, indices: torch.Tensor) -> "MatrixStep":
        """Returns the step for some of the elements."""

        return MatrixStep(self.matrices[indices], self.input_shape, self.output_shape)


@dataclass
class WarpStep:
    """A step of a warp, which keeps the canvas."""

    backward_function: WarpFunction  # Maps coordinates after the warp to coordinates before it.
    forward_function: WarpFunction  # Maps coordinates before the warp to coordinates after it.
    parameters: dict[str, torch.Tensor]  # Per-element parameters, float64 on the CPU.
    mask: torch.Tensor | None  # Which elements are warped, of shape `(B,)`, or `None` for all.
    shape: tuple[int, ...]  # Spatial shape of the canvas.
    grid_step: float  # Spacing of the grid the warp is evaluated on, in pixels.
    _cache: dict[tuple[torch.device, torch.dtype], dict[str, torch.Tensor]] = field(
        default_factory=dict, repr=False
    )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.shape

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Moves coordinates of shape `(B, ..., n)` through the warp."""

        return self._call(self.forward_function, coordinates)

    def backward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Moves coordinates of shape `(B, ..., n)` back through the warp."""

        return self._call(self.backward_function, coordinates)

    def inverted(self) -> "WarpStep":
        """Returns the step that undoes this one."""

        return WarpStep(
            self.forward_function,
            self.backward_function,
            self.parameters,
            self.mask,
            self.shape,
            self.grid_step,
        )

    def subset(self, indices: torch.Tensor) -> "WarpStep":
        """Returns the step for some of the elements."""

        return WarpStep(
            self.backward_function,
            self.forward_function,
            {name: value[indices] for name, value in self.parameters.items()},
            self.mask[indices] if self.mask is not None else None,
            self.shape,
            self.grid_step,
        )

    def warped(self, batch_size: int) -> torch.Tensor:
        """Returns which elements the warp moves, of shape `(B,)`."""

        if self.mask is None:
            return torch.ones(batch_size, dtype=torch.bool)

        return self.mask

    def _call(self, function: WarpFunction, coordinates: torch.Tensor) -> torch.Tensor:
        parameters = self._get_parameters(coordinates.device, coordinates.dtype)
        moved = function(coordinates, parameters)

        if self.mask is None:
            return moved

        mask = self.mask.to(device=coordinates.device).view(-1, *(1,) * (coordinates.ndim - 1))

        return torch.where(mask, moved, coordinates)

    def _get_parameters(self, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """Returns the parameters on a device, floating point ones in a dtype."""

        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = {
                name: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
                for name, value in self.parameters.items()
            }

        return self._cache[key]


# A step of a chain.
Step = MatrixStep | WarpStep


def size_of(shape: tuple[int, ...]) -> torch.Tensor:
    """Returns the size of a spatial shape in `(x, y[, z])` order, as a float64 tensor."""

    return torch.tensor(shape[::-1], dtype=torch.float64)


class Sampling:
    """How to get the output canvas from the input canvas, resolved from a chain of steps."""

    def __init__(
        self,
        steps: list[Step],
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_size: int,
    ) -> None:
        """Resolves a chain.

        Args:
            steps: The steps in the order they were applied.
            input_shape: Spatial shape of the canvas before the first step.
            output_shape: Spatial shape of the canvas after the last step.
            batch_size: Batch size B.
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.ndim = len(input_shape)

        # Multiply every run of consecutive matrices into one.
        self._items: list[torch.Tensor | WarpStep] = []
        for step in steps:
            if isinstance(step, MatrixStep):
                if self._items and isinstance(self._items[-1], torch.Tensor):
                    self._items[-1] = step.matrices @ self._items[-1]
                else:
                    self._items.append(step.matrices)
            else:
                self._items.append(step)

        self._warps = [x for x in self._items if isinstance(x, WarpStep)]

        # The product of all matrices, ignoring warps, as a backward matrix in pixel coordinates.
        product = mat.identity(batch_size, self.ndim)
        for item in self._items:
            if isinstance(item, torch.Tensor):
                product = item @ product

        self._centered_backward = torch.linalg.inv(product)
        self._backward = mat.centered_to_pixel(
            self._centered_backward, size_of(output_shape), size_of(input_shape)
        )

        self._steps = steps
        self.kind = self._get_kind()

        # Grids by device and dtype.
        self._grids: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _get_kind(self) -> str:
        """Returns `"identity"`, `"gather"`, `"mixed"`, `"affine"`, `"projective"` or `"warp"`.

        Elements that are only flipped, permuted and moved by whole pixels, and not warped, are
        exact: they are gathered. When only some elements are, the batch is split in two.
        """

        ndim = self.ndim
        permutations, signs, exact = mat.signed_permutations(self._backward)

        for warp in self._warps:
            exact = exact & ~warp.warped(self.batch_size)

        # A pixel center at `i + 0.5` must land on a pixel center.
        offsets = self._backward[:, :ndim, ndim] + 0.5 * signs - 0.5
        exact = exact & (torch.abs(offsets - torch.round(offsets)) < mat.INTEGER_TOLERANCE).all(
            dim=1
        )

        if bool(exact.all()):
            if self.input_shape == self.output_shape:
                identity = mat.identity(self.batch_size, ndim)
                if bool(torch.all(torch.abs(self._backward - identity) < mat.INTEGER_TOLERANCE)):
                    return "identity"

            self._permutations = permutations
            self._signs = signs.to(dtype=torch.int64)
            self._offsets = torch.round(offsets).to(dtype=torch.int64)

            return "gather"

        if bool(exact.any()):
            self._exact_indices = torch.nonzero(exact)[:, 0]
            self._other_indices = torch.nonzero(~exact)[:, 0]

            self._exact = self._subset(self._exact_indices)
            self._other = self._subset(self._other_indices)

            return "mixed"

        if self._warps:
            return "warp"

        if mat.is_projective(self._backward):
            return "projective"

        return "affine"

    def _subset(self, indices: torch.Tensor) -> "Sampling":
        """Returns the sampling of some of the elements."""

        return Sampling(
            [x.subset(indices) for x in self._steps],
            self.input_shape,
            self.output_shape,
            len(indices),
        )

    def apply(self, target: RasterTarget, data: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Resamples data of a raster target.

        Args:
            target: The target, for its interpolation and padding.
            data: Its current data, of shape `(B, C, *input_shape)`, floating point.

        Returns:
            The data on the output canvas, and whether it is a new tensor rather than (a view of)
            the given one.
        """

        if self.kind == "identity":
            return data, False

        if self.kind == "gather":
            return self._gather(target, data)

        if self.kind == "mixed":
            output = data.new_empty((self.batch_size, data.shape[1], *self.output_shape))

            for indices, sampling in (
                (self._exact_indices, self._exact),
                (self._other_indices, self._other),
            ):
                device_indices = indices.to(data.device)
                output[device_indices] = sampling.apply(target, data[device_indices])[0]

            return output, True

        return self._sample(target, data), True

    def footprints(self) -> torch.Tensor:
        """Returns how many input pixels an output pixel spans per input axis, of shape `(B, n)`."""

        linear = self._backward[:, : self.ndim, : self.ndim]

        return torch.abs(linear).sum(dim=2)

    def _gather(self, target: RasterTarget, data: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Resamples by gathering whole pixels, for signed permutations and whole-pixel moves."""

        ndim = self.ndim
        batch_size = data.shape[0]
        device = data.device

        # From here on, axes are in tensor order, the reverse of coordinate order.
        input_shape, output_shape = self.input_shape, self.output_shape

        permutations = (ndim - 1 - self._permutations).flip(1)
        signs = self._signs.flip(1)
        offsets = self._offsets.flip(1)

        # A uniform batch of whole-pixel crops, flips and transposes that stays within the input is
        # just a view, flipped where needed.
        uniform = bool(
            torch.all(permutations == permutations[0])
            and torch.all(signs == signs[0])
            and torch.all(offsets == offsets[0])
        )

        if uniform:
            view = self._view(data, permutations[0], signs[0], offsets[0])
            if view is not None:
                return view

        # Output axis `j` is read by the input axis whose row has its entry in column `j`. Since
        # every row has a single entry, per output axis `j` that input axis is found by inverting
        # the permutation.
        inverse_permutations = torch.argsort(permutations, dim=1)

        output = None
        for permutation in torch.unique(inverse_permutations, dim=0):
            indices = torch.nonzero((inverse_permutations == permutation).all(dim=1))[:, 0]
            permuted = data[indices.to(device)].permute(0, 1, *(2 + permutation).tolist())

            # The index along every output axis, of shape (b, S_j), and whether it is valid.
            axis_indices = []
            axis_valid = []
            for j in range(ndim):
                k = int(permutation[j])
                size = input_shape[k]

                positions = torch.arange(output_shape[j], dtype=torch.int64)
                index = signs[indices, k, None] * positions + offsets[indices, k, None]

                valid = (index >= 0) & (index < size)
                if target.padding == "reflection":
                    index = _reflect(index, size)
                else:
                    index = index.clamp(0, size - 1)

                axis_indices.append(index)
                axis_valid.append(valid)

            # Linear indices into the flattened spatial dimensions, broadcast over all axes.
            linear = torch.zeros((len(indices),) + (1,) * ndim, dtype=torch.int64)
            valid = torch.ones((len(indices),) + (1,) * ndim, dtype=torch.bool)
            stride = 1
            for j in reversed(range(ndim)):
                shape = [len(indices)] + [1] * ndim
                shape[1 + j] = output_shape[j]

                linear = linear + axis_indices[j].view(shape) * stride
                valid = valid & axis_valid[j].view(shape)
                stride *= permuted.shape[2 + j]

            linear = (
                linear.expand(len(indices), *output_shape).reshape(len(indices), 1, -1).to(device)
            )
            channels = permuted.shape[1]

            gathered = torch.gather(
                permuted.reshape(len(indices), channels, -1), 2, linear.expand(-1, channels, -1)
            ).view(len(indices), channels, *output_shape)

            if target.padding not in ("border", "reflection"):
                fill = _get_fill(target, data[indices.to(device)])
                valid = valid.to(device)[:, None]
                gathered = torch.where(valid, gathered, fill)

            if len(indices) == batch_size:
                return gathered, True

            if output is None:
                output = data.new_empty((batch_size, channels, *output_shape))

            output[indices.to(device)] = gathered

        return output, True

    def _view(
        self,
        data: torch.Tensor,
        permutation: torch.Tensor,
        signs: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, bool] | None:
        """Returns the output as a view of the input, if it lies within it."""

        ndim = self.ndim
        inverse = torch.argsort(permutation).tolist()
        output = data.permute(0, 1, *(2 + x for x in inverse))

        slices = [slice(None), slice(None)]
        flips = []
        for j in range(ndim):
            k = inverse[j]
            size = self.input_shape[k]
            count = self.output_shape[j]
            sign, offset = int(signs[k]), int(offsets[k])

            first, last = (offset, offset + count - 1) if sign > 0 else (offset - count + 1, offset)
            if first < 0 or last >= size:
                return None

            slices.append(slice(first, last + 1))
            if sign < 0:
                flips.append(2 + j)

        output = output[tuple(slices)]

        if flips:
            return torch.flip(output, flips), True

        return output, False

    def _sample(self, target: RasterTarget, data: torch.Tensor) -> torch.Tensor:
        """Resamples with `grid_sample`, from a mipmap level per element where it shrinks."""

        grid = self._get_grid(data.device, data.dtype)

        levels = torch.zeros(self.batch_size, self.ndim, dtype=torch.int64)
        if target.antialias and target.mode != "nearest":
            levels = self._get_levels()

        fill = None
        if target.padding not in ("zeros", "border", "reflection"):
            fill = _get_fill(target, data)

        unique_levels = torch.unique(levels, dim=0)
        if len(unique_levels) == 1:
            source = self._get_level(data, tuple(unique_levels[0].tolist()), {})

            return self._sample_level(target, source, grid, fill)

        # Level zero is sampled for the whole batch, which is cheaper than gathering the elements
        # at that level from the full-size input. Coarser levels are pooled for the whole batch,
        # which reads the input once, and only their small pooled images are gathered.
        cache: dict[tuple[int, ...], torch.Tensor] = {}
        zero = (0,) * self.ndim
        at_zero = (levels == 0).all(dim=1)

        if bool(at_zero.any()):
            output = self._sample_level(target, data, grid, fill)
        else:
            output = data.new_empty((self.batch_size, data.shape[1], *self.output_shape))

        for level in unique_levels:
            key = tuple(level.tolist())
            if key == zero:
                continue

            indices = torch.nonzero((levels == level).all(dim=1))[:, 0].to(data.device)
            source = self._get_level(data, key, cache)[indices]
            element_fill = fill[indices] if fill is not None and fill.shape[0] > 1 else fill

            output[indices] = self._sample_level(target, source, grid[indices], element_fill)

        return output

    def _get_levels(self) -> torch.Tensor:
        """Returns the mipmap level per element and axis, in tensor order, of shape `(B, n)`.

        An element shrunk about evenly along all axes (within a factor of two) gets one level for
        all of them, from the axis it shrinks most along, so it needs a single mipmap. Only an
        element shrunk much more along one axis than another gets a level per axis.
        """

        footprints = self.footprints().flip(1).clamp(min=1)
        levels = torch.round(torch.log2(footprints)).to(dtype=torch.int64)

        largest = footprints.amax(dim=1, keepdim=True)
        even = largest <= 2 * footprints.amin(dim=1, keepdim=True)
        uniform = torch.round(torch.log2(largest)).to(dtype=torch.int64).expand_as(levels)

        return torch.where(even, uniform, levels)

    def _get_level(
        self, data: torch.Tensor, level: tuple[int, ...], cache: dict[tuple[int, ...], torch.Tensor]
    ) -> torch.Tensor:
        """Returns data shrunk by two to the power of a level per axis, averaging pixels.

        The levels are built by halving, one axis and level at a time. On an even size, linear
        interpolation to half the size samples exactly between every two pixels, so it averages
        them, and is the fastest way to do so. An odd size is pooled adaptively. Both keep the
        extent of the data, so normalized sampling coordinates stay valid.
        """

        current = (0,) * self.ndim
        output = data

        while current != level:
            # One axis at a time, so levels that differ along one axis share the others.
            axis = next(a for a in range(self.ndim) if current[a] < level[a])
            step = tuple(c + 1 if a == axis else c for a, c in enumerate(current))

            if step not in cache:
                size = output.shape[2 + axis]

                if size % 2 == 0:
                    mode = "bilinear" if self.ndim == 2 else "trilinear"
                    scale = tuple(0.5 if a == axis else 1.0 for a in range(self.ndim))
                    cache[step] = F.interpolate(
                        output, scale_factor=scale, mode=mode, align_corners=False
                    )
                else:
                    pool = F.adaptive_avg_pool2d if self.ndim == 2 else F.adaptive_avg_pool3d
                    shape = list(output.shape[2:])
                    shape[axis] = max(1, math.ceil(size / 2))
                    cache[step] = pool(output, tuple(shape))

            output = cache[step]
            current = step

        return output

    def _sample_level(
        self,
        target: RasterTarget,
        source: torch.Tensor,
        grid: torch.Tensor,
        fill: torch.Tensor | None,
    ) -> torch.Tensor:
        """Samples a source on a grid, filling where it falls outside it."""

        padding_mode = target.padding if target.padding in ("border", "reflection") else "zeros"
        output = F.grid_sample(
            source, grid, mode=target.mode, padding_mode=padding_mode, align_corners=False
        )

        if fill is None:
            return output

        # Sampling with zeros around the source leaves out exactly the part of the fill that lies
        # outside it, which is its coverage of each output pixel.
        coverage = _get_coverage(grid, source.shape[2:], target.mode)

        return output.add_((1 - coverage).unsqueeze(1) * fill)

    def _get_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns the sampling grid, of shape `(B, *output_shape, n)` in normalized coordinates."""

        key = (device, dtype)
        if key not in self._grids:
            if self.kind == "affine":
                self._grids[key] = self._get_affine_grid(device, dtype)
            elif self.kind == "projective":
                self._grids[key] = self._get_projective_grid(device, dtype)
            else:
                self._grids[key] = self._get_warp_grid(device, dtype)

        return self._grids[key]

    def _get_affine_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ndim = self.ndim

        # From normalized output coordinates to normalized input coordinates.
        to_output = mat.scale((0.5 * size_of(self.output_shape)).expand(self.batch_size, -1))
        to_input = mat.scale((2 / size_of(self.input_shape)).expand(self.batch_size, -1))
        theta = to_input @ self._centered_backward @ to_output

        return F.affine_grid(
            theta[:, :ndim].to(device=device, dtype=dtype),
            [self.batch_size, 1, *self.output_shape],
            align_corners=False,
        )

    def _get_projective_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        centers = _get_centers(self.output_shape, device, dtype)
        coordinates = mat.apply(
            self._centered_backward, centers.expand(self.batch_size, *centers.shape[1:])
        )

        return coordinates * (2 / size_of(self.input_shape)).to(device=device, dtype=dtype)

    def _get_warp_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_step = min(x.grid_step for x in self._warps)

        # A coarse lattice that includes the first and last pixel centers, so upsampling it with
        # `align_corners=True` lands exactly on every pixel center, and affine parts stay exact.
        coarse_shape = tuple(
            (
                size
                if size <= 2 or grid_step <= 1
                else min(size, max(2, math.ceil((size - 1) / grid_step) + 1))
            )
            for size in self.output_shape
        )
        coordinates = _get_centers(self.output_shape, device, dtype, coarse_shape)
        coordinates = coordinates.expand(self.batch_size, *coordinates.shape[1:])

        for item in reversed(self._items):
            if isinstance(item, torch.Tensor):
                coordinates = mat.apply(torch.linalg.inv(item), coordinates)
            else:
                coordinates = item.backward(coordinates)

        coordinates = coordinates * (2 / size_of(self.input_shape)).to(device=device, dtype=dtype)

        if coarse_shape != self.output_shape:
            coordinates = torch.movedim(coordinates, -1, 1)
            coordinates = F.interpolate(
                coordinates,
                size=self.output_shape,
                mode="bilinear" if self.ndim == 2 else "trilinear",
                align_corners=True,
            )
            coordinates = torch.movedim(coordinates, 1, -1)

        return coordinates.contiguous()


def _get_centers(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    lattice_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Returns pixel centers of a canvas in centered coordinates, of shape `(1, *lattice_shape, n)`.

    With a lattice shape smaller than the canvas, the centers are spread evenly from the first to
    the last pixel center.
    """

    lattice_shape = lattice_shape or shape

    axes = [
        torch.linspace(-0.5 * size + 0.5, 0.5 * size - 0.5, count, device=device, dtype=dtype)
        for size, count in zip(shape, lattice_shape)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")

    return torch.stack(grids[::-1], dim=-1)[None]


def _get_coverage(grid: torch.Tensor, shape: tuple[int, ...], mode: str) -> torch.Tensor:
    """Returns how much of every sample on a grid lies within the source, of shape `(B, *S)`.

    With zeros padding, a bilinear sample at pixel position `x` (pixel `i` at `i`) of an axis of
    `N` pixels takes weight from within the source of `clamp(min(1 + x, N - x), 0, 1)`, and the
    weights of the axes multiply. Bicubic samples are taken to cover the same.
    """

    coverage = None
    for axis, size in enumerate(shape[::-1]):
        position = ((grid[..., axis] + 1) * size - 1) / 2

        if mode == "nearest":
            axis_coverage = ((position > -0.5) & (position < size - 0.5)).to(dtype=grid.dtype)
        else:
            axis_coverage = torch.minimum(1 + position, size - position).clamp_(0, 1)

        coverage = axis_coverage if coverage is None else coverage * axis_coverage

    return coverage


def _reflect(index: torch.Tensor, size: int) -> torch.Tensor:
    """Reflects indices into `[0, size)`, repeating the edge, as `grid_sample`'s reflection does."""

    period = 2 * size
    index = torch.remainder(index, period)

    return torch.where(index >= size, period - 1 - index, index)


def _get_fill(target: RasterTarget, data: torch.Tensor) -> torch.Tensor:
    """Returns the fill value of a target, of shape `(B or 1, C, 1, ...)`."""

    ndim = data.ndim - 2
    spatial = tuple(range(2, data.ndim))

    if target.padding == "zeros":
        return torch.zeros((1, data.shape[1]) + (1,) * ndim, device=data.device, dtype=data.dtype)

    if target.padding == "mean":
        return data.mean(dim=spatial, keepdim=True)

    if target.padding == "median":
        # The median of a thinned grid of pixels is the same color at a fraction of the work.
        thinned = data[(slice(None), slice(None)) + (slice(None, None, 4),) * ndim]

        return (
            thinned.flatten(2).median(dim=2).values.view(data.shape[0], data.shape[1], *(1,) * ndim)
        )

    value = torch.as_tensor(target.padding, device=data.device, dtype=data.dtype).flatten()
    if len(value) == 1:
        value = value.expand(data.shape[1])

    if len(value) != data.shape[1]:
        raise ValueError(f"The fill value has {len(value)} values for {data.shape[1]} channels.")

    return value.view(1, -1, *(1,) * ndim)
