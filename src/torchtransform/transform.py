"""Base classes of transforms, and the transforms that combine others.

Every transform is a `torch.nn.Module`. Calling one augments a batch:

```python
augment = tt.Compose(tt.Rotate((-10, 10)), tt.Brightness((0.8, 1.2)))

images = augment(images)
outputs = augment(image=tt.Image(images), mask=tt.Mask(masks), boxes=tt.Boxes(boxes))
```

Every batch element draws its own parameters and decides on its own whether a transform with a
probability, or a branch of `OneOf` or `SomeOf`, applies to it.

To write a transform, subclass one of:

- `MatrixTransform`, for a transform given by a matrix per element (affine or projective).
- `CanvasTransform`, for a matrix that also changes the shape of the canvas (crops, pads, resizes).
- `WarpTransform`, for a smooth nonlinear map of coordinates.
- `ColorTransform`, for a transform that is affine per pixel in color (a matrix per element).
- `PixelTransform`, for anything else that changes pixels.

The first three are fused into a single resampling, and consecutive color transforms into a single
matrix product.
"""

from typing import Any, TypeVar

import torch

from torchtransform import matrices as mat
from torchtransform.distributions import Parameter, to_distribution
from torchtransform.resampling import MatrixStep, WarpStep
from torchtransform.state import (
    ColorView,
    Context,
    State,
    collect_targets,
    identity_where,
    next_transform_seed,
)

# A transform, for methods that return themselves.
TransformType = TypeVar("TransformType", bound="Transform")

# Most iterations of the fixed-point inversion of a warp, and the error in pixels it stops at.
WARP_INVERSION_ITERATIONS: int = 40
WARP_INVERSION_TOLERANCE: float = 1e-7


class Transform(torch.nn.Module):
    """Base class of all transforms."""

    def __init__(self, p: float = 1.0) -> None:
        """Initializes the transform.

        Args:
            p: Probability that the transform is applied, drawn for every batch element.
        """

        super().__init__()

        if not 0 <= p <= 1:
            raise ValueError(f"The probability must be in [0, 1], not {p}.")

        self.p = float(p)
        self._seed = next_transform_seed()

    @property
    def seed(self) -> int:
        """The seed of the transform. Its draws depend on it and on the seed of the call."""

        return self._seed

    def manual_seed(self: TransformType, seed: int) -> TransformType:
        """Sets the seed of the transform.

        Two transforms with the same seed and parameters draw the same values in calls with the
        same seed.

        Args:
            seed: The seed.

        Returns:
            The transform itself.
        """

        self._seed = int(seed)

        return self

    def forward(
        self,
        *inputs: Any,
        seed: int | None = None,
        shape: tuple[int, ...] | None = None,
        replay: bool = False,
        **targets: Any,
    ) -> Any:
        """Transforms a batch.

        Args:
            inputs: Tensors (taken as images) or targets, by position.
            seed: Seed of the call. Drawn from torch's global generator if not given, so
                `torch.manual_seed` and data loader worker seeding make calls reproducible.
            shape: Spatial shape of the canvas, only needed when there are no images or masks.
            replay: Whether to also return a `Replay` of the geometric transforms.
            targets: Tensors (taken as images) or targets, by name.

        Returns:
            For a single positional input, its output. For several, a tuple of outputs. For keyword
            inputs, a dictionary of outputs by name. With `replay`, a tuple of that and the replay.
        """

        all_targets, pack = collect_targets(inputs, targets)
        state = State(all_targets, seed=seed, shape=shape)

        self.run(state, None, False)
        outputs = pack(state.finish())

        if replay:
            return outputs, state.replay()

        return outputs

    def run(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        """Applies the transform to the elements in a mask, drawing its own probability.

        Args:
            state: The state of the batch.
            mask: Which elements to apply to, of shape `(B,)`, or `None` for all.
            inverse: Whether to apply the inverse.
        """

        mask = state.probability_mask(self, mask)
        if mask is not None and not bool(mask.any()):
            return

        if mask is not None and bool(mask.all()):
            mask = None

        self.apply(state, mask, inverse)

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        """Applies the transform. Called by `run` once the elements it applies to are known.

        Args:
            state: The state of the batch.
            mask: Which elements to apply to, of shape `(B,)`, or `None` for all.
            inverse: Whether to apply the inverse.
        """

        raise NotImplementedError()

    def extra_repr(self) -> str:
        return f"p={self.p}" if self.p < 1 else ""


class Compose(Transform):
    def __init__(self, *transforms: Transform, p: float = 1.0) -> None:
        """Applies transforms one after the other.

        Args:
            transforms: The transforms.
            p: Probability that the whole sequence is applied, per element.
        """

        super().__init__(p)

        self.transforms = torch.nn.ModuleList(transforms)

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        transforms = reversed(self.transforms) if inverse else self.transforms

        for transform in transforms:
            transform.run(state, mask, inverse)


class Maybe(Compose):
    def __init__(self, transform: Transform, p: float = 0.5) -> None:
        """Applies a transform with a probability, per element.

        Args:
            transform: The transform.
            p: Probability that it is applied.
        """

        super().__init__(transform, p=p)


class OneOf(Transform):
    def __init__(
        self, *transforms: Transform, weights: tuple[float, ...] | None = None, p: float = 1.0
    ) -> None:
        """Applies one of several transforms, chosen per element.

        Args:
            transforms: The transforms.
            weights: Relative probability of every transform. Equal if not given.
            p: Probability that any of them is applied.
        """

        super().__init__(p)

        if weights is not None and len(weights) != len(transforms):
            raise ValueError("OneOf needs as many weights as transforms.")

        self.transforms = torch.nn.ModuleList(transforms)
        self.weights = tuple(weights) if weights is not None else (1.0,) * len(transforms)

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        if not self.transforms:
            return

        def draw() -> torch.Tensor:
            context = state.context(self)
            weights = torch.tensor(self.weights, dtype=torch.float64)

            return torch.multinomial(
                weights, state.batch_size, replacement=True, generator=context.generator
            )

        choices = state.cached(self, "choices", draw)

        for i, transform in enumerate(self.transforms):
            chosen = choices == i
            transform.run(state, chosen if mask is None else mask & chosen, inverse)


class SomeOf(Transform):
    def __init__(
        self, *transforms: Transform, n: int | tuple[int, int] = 1, p: float = 1.0
    ) -> None:
        """Applies some of several transforms, chosen per element, in the order they are given.

        Args:
            transforms: The transforms.
            n: How many to apply, or a `(low, high)` range to draw it from, both bounds included.
            p: Probability that any of them is applied.
        """

        super().__init__(p)

        low, high = (n, n) if isinstance(n, int) else n
        if not 0 <= low <= high <= len(transforms):
            raise ValueError(f"SomeOf cannot apply {n} of {len(transforms)} transforms.")

        self.transforms = torch.nn.ModuleList(transforms)
        self.n = (low, high)

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        if not self.transforms:
            return

        def draw() -> torch.Tensor:
            context = state.context(self)
            counts = torch.randint(
                self.n[0], self.n[1] + 1, (state.batch_size, 1), generator=context.generator
            )

            # A random rank per transform and element; the lowest ranks are chosen.
            ranks = torch.argsort(context.rand(state.batch_size, len(self.transforms)), dim=1)
            ranks = torch.argsort(ranks, dim=1)

            return ranks < counts

        chosen = state.cached(self, "chosen", draw)

        order = reversed(range(len(self.transforms))) if inverse else range(len(self.transforms))
        for i in order:
            selected = chosen[:, i]
            self.transforms[i].run(state, selected if mask is None else mask & selected, inverse)


class Inverse(Transform):
    def __init__(self, transform: Transform, p: float = 1.0) -> None:
        """Applies the inverse of a transform.

        Within one call, the inverse undoes exactly what the transform drew, so
        `Compose(t, x, Inverse(t))` applies `x` in the frame of `t`. To undo the geometry of a
        call afterward, such as for test-time augmentation, use its `Replay` instead.

        Args:
            transform: The transform to invert.
            p: Probability that the inverse is applied.
        """

        super().__init__(p)

        self.transform = transform

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        self.transform.run(state, mask, not inverse)


class Identity(Transform):
    def __init__(self) -> None:
        """Does nothing. Useful as a branch of `OneOf`."""

        super().__init__()

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        pass


class MatrixTransform(Transform):
    """A geometric transform given by a matrix per element, fused with its neighbors.

    Subclasses implement `get_matrices`. Matrices work in centered coordinates: pixels, with the
    origin in the center of the canvas.
    """

    def get_matrices(self, context: Context) -> torch.Tensor:
        """Returns the matrices of the transform.

        Args:
            context: The context, to draw from and to look at the canvas and geometry.

        Returns:
            Forward matrices of shape `(B, n + 1, n + 1)`, float64 on the CPU, that map coordinates
            before the transform to coordinates after it.
        """

        raise NotImplementedError()

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        # Cached for the whole call, so an inverse after a change of canvas undoes exactly these.
        matrices = state.cached(self, "matrices", lambda: self.get_matrices(state.context(self)))

        if inverse:
            matrices = torch.linalg.inv(matrices)

        state.push(MatrixStep(identity_where(mask, matrices), state.shape, state.shape))


class CanvasTransform(Transform):
    """A geometric transform that also changes the shape of the canvas.

    The canvas is shared by the whole batch, so a canvas transform changes it for every element.
    Elements it does not apply to (by its probability, or by not being chosen in `OneOf` or
    `SomeOf`) are left unscaled in the center of the new canvas.

    Subclasses implement `get_output_shape` and `get_matrices`.
    """

    def get_output_shape(self, context: Context) -> tuple[int, ...]:
        """Returns the spatial shape of the canvas after the transform.

        Args:
            context: The context, whose canvas is the one before the transform.
        """

        raise NotImplementedError()

    def get_matrices(self, context: Context, output_shape: tuple[int, ...]) -> torch.Tensor:
        """Returns the matrices of the transform.

        Args:
            context: The context, whose canvas is the one before the transform.
            output_shape: The shape of the canvas after it.

        Returns:
            Forward matrices of shape `(B, n + 1, n + 1)` in centered coordinates, float64 on the
            CPU, from the canvas before the transform to the canvas after it.
        """

        raise NotImplementedError()

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        if inverse:
            record = state.records.get(id(self))
            if record is None:
                raise RuntimeError(
                    f"{type(self).__name__} can only be inverted after it was applied in the same "
                    f"call. Use a `Replay` to undo it afterward."
                )

            matrices, input_shape, output_shape = record
            if state.shape != output_shape:
                raise RuntimeError(
                    f"{type(self).__name__} made a canvas of {output_shape}, not {state.shape}."
                )

            inverse_matrices = torch.linalg.inv(matrices)
            if mask is not None:
                neutral = centered_placement(state.batch_size, output_shape, input_shape)
                inverse_matrices = torch.where(mask[:, None, None], inverse_matrices, neutral)

            state.push(MatrixStep(inverse_matrices, output_shape, input_shape))

            return

        input_shape = state.shape
        output_shape = tuple(int(x) for x in self.get_output_shape(state.context(self)))

        matrices = state.cached(
            self,
            ("matrices", input_shape),
            lambda: self.get_matrices(state.context(self), output_shape),
        )

        if mask is not None:
            neutral = centered_placement(state.batch_size, input_shape, output_shape)
            matrices = torch.where(mask[:, None, None], matrices, neutral)

        state.record(self, (matrices, input_shape, output_shape))
        state.push(MatrixStep(matrices, input_shape, output_shape))


class WarpTransform(Transform):
    """A smooth nonlinear map of coordinates, fused with its neighbors.

    Subclasses implement `get_parameters` and `backward_coordinates`, and `forward_coordinates` if
    they have it in closed form. Pixels need the backward map, from coordinates after the warp to
    where they come from; geometry needs the forward map, which by default inverts the backward
    map numerically.

    Warps are evaluated on a grid with a spacing of `grid_step` pixels, and interpolated in between.
    """

    def __init__(self, grid_step: float = 4.0, p: float = 1.0) -> None:
        """Initializes the warp.

        Args:
            grid_step: Spacing of the grid the warp is evaluated on, in pixels. One evaluates it at
                every pixel.
            p: Probability that the warp is applied, per element.
        """

        super().__init__(p)

        if grid_step < 1:
            raise ValueError("The grid step must be at least one pixel.")

        self.grid_step = float(grid_step)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        """Returns the parameters of the warp.

        Args:
            context: The context, to draw from and to look at the canvas.

        Returns:
            Parameters by name, every one a float64 CPU tensor with the batch as its first
            dimension.
        """

        return {}

    def backward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Maps coordinates after the warp to coordinates before it.

        Args:
            coordinates: Centered coordinates of shape `(B, ..., n)`.
            parameters: The parameters, on the device and in the dtype of the coordinates.

        Returns:
            The coordinates before the warp, of the same shape.
        """

        raise NotImplementedError()

    def forward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Maps coordinates before the warp to coordinates after it.

        By default this inverts `backward_coordinates` by fixed-point iteration, which converges for
        any warp that moves nearby points by amounts that differ less than their distance.

        Args:
            coordinates: Centered coordinates of shape `(B, ..., n)`.
            parameters: The parameters, on the device and in the dtype of the coordinates.

        Returns:
            The coordinates after the warp, of the same shape.
        """

        moved = coordinates
        for iteration in range(WARP_INVERSION_ITERATIONS):
            error = coordinates - self.backward_coordinates(moved, parameters)
            moved = moved + error

            # Checking every few iterations keeps the synchronizations on a GPU few.
            if iteration % 4 == 3 and bool(error.abs().max() < WARP_INVERSION_TOLERANCE):
                break

        return moved

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        # Cached for the whole call, so an inverse after a change of canvas undoes exactly these.
        parameters = state.cached(
            self, "parameters", lambda: self.get_parameters(state.context(self))
        )

        step = WarpStep(
            self.backward_coordinates,
            self.forward_coordinates,
            parameters,
            mask,
            state.shape,
            self.grid_step,
        )

        state.push(step.inverted() if inverse else step)

    def extra_repr(self) -> str:
        return ", ".join(x for x in (super().extra_repr(), f"grid_step={self.grid_step}") if x)


class ColorTransform(Transform):
    """A transform that is affine per pixel in color, fused with its neighbors.

    Subclasses implement `get_color_matrices`. Consecutive color transforms are multiplied into a
    single matrix per element, applied in one pass, with values clamped to `[0, 1]` once at the
    end rather than after every one of them.
    """

    # Whether the result is clamped to [0, 1].
    clamp: bool = True

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        """Returns the color matrices of the transform for an image.

        Args:
            context: The context, to draw from. It draws the same values for every image.
            image: The image, for its number of channels and its statistics.

        Returns:
            Matrices of shape `(B, C + 1, C + 1)`, float64 on the CPU, that map a color `(c, 1)`.
        """

        raise NotImplementedError()

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        for name, target in state.raster.items():
            if not target.photometric:
                continue

            # Cached, so an inverse later in the call undoes exactly these, even when they depend
            # on the image, which will have changed by then.
            matrices = state.cached(
                self,
                ("color", name),
                lambda: self.get_color_matrices(state.context(self), ColorView(state, name)),
            )

            if inverse:
                matrices = torch.linalg.inv(matrices)

            state.push_color(name, identity_where(mask, matrices), self.clamp, mask)


class PixelTransform(Transform):
    """A transform that changes the pixels of images.

    Subclasses implement `apply_image`, and `get_parameters` to draw parameters per element. The
    parameters are drawn for the whole batch, so an element's parameters do not depend on which
    other elements the transform applies to.
    """

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        """Returns the parameters of the transform.

        Args:
            context: The context, to draw from and to look at the canvas.

        Returns:
            Parameters by name, every one a CPU tensor with the batch as its first dimension.
        """

        return {}

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        """Transforms images.

        Args:
            image: The images the transform applies to, of shape `(b, C, ...)`, floating point. It
                may be changed in-place.
            parameters: The parameters of those images, floating point ones on their device and in
                their dtype, and the others on their device.
            context: The context, to draw noise with `context.device_generator()`.

        Returns:
            The transformed images, of the same shape.
        """

        raise NotImplementedError()

    def apply(self, state: State, mask: torch.Tensor | None, inverse: bool) -> None:
        if inverse:
            raise NotImplementedError(f"{type(self).__name__} cannot be inverted.")

        state.resolve()

        parameters = state.cached(
            self, "parameters", lambda: self.get_parameters(state.context(self))
        )
        indices = torch.nonzero(mask)[:, 0] if mask is not None else None

        for name, target in state.raster.items():
            if not target.photometric:
                continue

            data = state.data[name]
            selected = _select(parameters, indices, data)
            context = state.context(self, indices=indices)

            if indices is None:
                state.set(name, self.apply_image(state.writable(name), selected, context))
            else:
                device_indices = indices.to(data.device)
                output = self.apply_image(data[device_indices], selected, context)

                state.writable(name)[device_indices] = output


class Lambda(PixelTransform):
    def __init__(self, function: Any, p: float = 1.0) -> None:
        """Applies a function to images.

        Args:
            function: Takes images of shape `(b, C, ...)` and returns them transformed.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.function = function

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        return self.function(image)


def _select(
    parameters: dict[str, torch.Tensor], indices: torch.Tensor | None, data: torch.Tensor
) -> dict[str, Any]:
    """Selects the parameters of some elements and moves them to the data."""

    selected: dict[str, Any] = {}
    for name, value in parameters.items():
        if isinstance(value, torch.Tensor):
            if indices is not None and value.ndim > 0:
                value = value[indices]

            value = value.to(
                device=data.device, dtype=data.dtype if value.is_floating_point() else None
            )

        selected[name] = value

    return selected


def centered_placement(
    batch_size: int, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> torch.Tensor:
    """Returns the matrices that put a canvas unscaled in the center of another, on whole pixels.

    When the sizes differ by an odd number of pixels, the content goes half a pixel toward the
    start, so pixels stay on pixels.

    Args:
        batch_size: Batch size B.
        input_shape: Spatial shape of the canvas before.
        output_shape: Spatial shape of the canvas after.

    Returns:
        Matrices of shape `(B, n + 1, n + 1)` in centered coordinates.
    """

    difference = torch.tensor(input_shape[::-1], dtype=torch.float64) - torch.tensor(
        output_shape[::-1], dtype=torch.float64
    )
    shift = 0.5 * difference - torch.floor(0.5 * difference)

    return mat.translation(shift.expand(batch_size, -1))


def sample_vector(
    context: Context, values: tuple[Parameter, ...], log: bool = False
) -> torch.Tensor:
    """Draws one parameter per spatial axis for every element, as a tensor of shape `(B, n)`."""

    return torch.stack(
        [
            to_distribution(x, log=log).sample((context.batch_size,), context.generator)
            for x in values
        ],
        dim=1,
    )
