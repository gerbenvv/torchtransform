"""The state of a batch while it is being transformed.

Geometric transforms do not touch pixels. They add steps to a pending chain, and move the geometry
targets (which is cheap) right away, so that transforms that depend on where the geometry is can
use it. Color transforms that are affine per pixel are also only accumulated, as one matrix per
element. Pending work is resolved when a transform needs pixels, or at the end, each run of it in a
single pass.
"""

import itertools
from collections.abc import Callable, Hashable
from typing import Any

import torch

from torchtransform import matrices as mat
from torchtransform.distributions import Parameter, to_distribution
from torchtransform.resampling import Sampling, Step, size_of
from torchtransform.targets import GeometryTarget, Image, RasterTarget, Target

# Mask for 64-bit seeds.
SEED_MASK: int = 0xFFFF_FFFF_FFFF_FFFF

# Counter that gives every transform its own seed, deterministically.
TRANSFORM_COUNTER: itertools.count = itertools.count(1)


def mix_seed(*values: int) -> int:
    """Mixes integers into a 64-bit seed with the SplitMix64 finalizer."""

    state = 0x9E37_79B9_7F4A_7C15
    for value in values:
        state = (state ^ (value & SEED_MASK)) & SEED_MASK
        state = ((state ^ (state >> 30)) * 0xBF58_476D_1CE4_E5B9) & SEED_MASK
        state = ((state ^ (state >> 27)) * 0x94D0_49BB_1331_11EB) & SEED_MASK
        state = state ^ (state >> 31)

    return state


def next_transform_seed() -> int:
    """Returns a seed for a newly created transform."""

    return mix_seed(next(TRANSFORM_COUNTER))


class Context:
    """What a transform can see and draw from while it is applied.

    All random draws of a transform in one call come from its own generators, seeded by the seed
    of the call and the seed of the transform. So a transform draws the same values however the
    rest of a pipeline is arranged, and `Inverse` of the same transform undoes exactly its draw.
    """

    def __init__(
        self,
        state: "State",
        transform: Any,
        inverse: bool,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Creates the context.

        Args:
            state: The state of the batch.
            transform: The transform being applied.
            inverse: Whether it is applied inversely.
            indices: For a pixel transform applied to part of the batch, the elements it is
                applied to.
        """

        self.state = state
        self.transform = transform
        self.inverse = inverse
        self.indices = indices

        self._seed = mix_seed(state.seed, transform.seed)
        self.generator = torch.Generator().manual_seed(mix_seed(self._seed, 1))

    @property
    def batch_size(self) -> int:
        """Batch size B."""

        return self.state.batch_size

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions, two or three."""

        return self.state.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape of the current canvas, such as `(H, W)`."""

        return self.state.shape

    @property
    def size(self) -> torch.Tensor:
        """Size of the current canvas in `(x, y[, z])` order, a float64 tensor of shape `(n,)`."""

        return size_of(self.state.shape)

    @property
    def device(self) -> torch.device:
        """Device of the data."""

        return self.state.device

    def sample(
        self, parameter: Parameter, *shape: int, log: bool = False, integer: bool = False
    ) -> torch.Tensor:
        """Draws a parameter for every batch element.

        Args:
            parameter: A number, a `(low, high)` tuple, a list of choices or a distribution.
            shape: Shape of the draw per element.
            log: Whether a range is drawn uniformly in the logarithm.
            integer: Whether a range is over integers, both bounds included.

        Returns:
            A float64 CPU tensor of shape `(B, *shape)`.
        """

        distribution = to_distribution(parameter, log=log, integer=integer)

        return distribution.sample((self.batch_size, *shape), self.generator)

    def rand(self, *shape: int) -> torch.Tensor:
        """Draws uniform values in `[0, 1)` as a float64 CPU tensor of the given shape."""

        return torch.rand(shape, dtype=torch.float64, generator=self.generator)

    def randn(self, *shape: int) -> torch.Tensor:
        """Draws standard normal values as a float64 CPU tensor of the given shape."""

        return torch.randn(shape, dtype=torch.float64, generator=self.generator)

    def randint(self, low: int, high: int, *shape: int) -> torch.Tensor:
        """Draws integers in `[low, high)` as an int64 CPU tensor of the given shape."""

        return torch.randint(low, high, shape, generator=self.generator)

    def device_generator(self, device: torch.device | None = None) -> torch.Generator:
        """Returns a generator on a device, for noise the size of the data.

        Every call returns a generator in the same state, so every target of a call gets the same
        noise.
        """

        device = torch.device(device) if device is not None else self.device

        return torch.Generator(device=device).manual_seed(mix_seed(self._seed, 2))

    def bounds(
        self, targets: tuple[Hashable, ...] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns where the geometry of every element currently is.

        Args:
            targets: The names of the geometry targets to consider, or `None` for all. An element
                without any points among them uses the corners of its content instead: the input
                canvas, moved by the transforms so far.

        Returns:
            The lowest and highest coordinates per element, each of shape `(B, n)`, as float64 in
            centered coordinates.
        """

        return self.state.bounds(targets)

    def points(
        self, targets: tuple[Hashable, ...] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the points of the geometry, as they currently are.

        Args:
            targets: The names of the geometry targets to select, or `None` for all. An element
                without any points among them selects the corners of its content instead.

        Returns:
            All points, of shape `(B, P, n)` as float64 in centered coordinates, and which of them
            are selected, of shape `(B, P)`.
        """

        return self.state.points, self.state.select(targets)

    def content_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns where the content of every element currently is: the moved input canvas."""

        return self.state.bounds(())

    def raster(self, name: Hashable) -> torch.Tensor | None:
        """Returns the current data of a raster target, for the elements being transformed.

        Args:
            name: Name of the target.

        Returns:
            Its data of shape `(b, C, ...)` in floating point, or `None` without such a target.
        """

        if name not in self.state.raster:
            return None

        self.state.resolve()
        data = self.state.data[name]

        if self.indices is not None:
            return data[self.indices.to(data.device)]

        return data


class ColorView:
    """The view a color transform has of an image while its matrix is being built."""

    def __init__(self, state: "State", name: Hashable) -> None:
        """Creates the view.

        Args:
            state: The state of the batch.
            name: Name of the image.
        """

        self.state = state
        self.name = name

    @property
    def channels(self) -> int:
        """Number of channels C."""

        return self.state.data[self.name].shape[1]

    def mean(self) -> torch.Tensor:
        """Returns the mean of every channel of every element as the image currently is.

        Pending color matrices are exact for means, so this only needs pending geometry resolved.

        Returns:
            A float64 CPU tensor of shape `(B, C)`.
        """

        self.state.resolve(geometry_only=True)

        data = self.state.data[self.name]
        means = data.mean(dim=tuple(range(2, data.ndim))).to(dtype=torch.float64).cpu()

        pending = self.state.pending_color(self.name)
        if pending is not None:
            homogeneous = torch.cat((means, torch.ones(len(means), 1, dtype=torch.float64)), dim=1)
            means = (pending @ homogeneous[..., None])[:, :-1, 0]

        return means

    def data(self) -> torch.Tensor:
        """Returns the image as it currently is, with everything pending resolved."""

        self.state.resolve()

        return self.state.data[self.name]


class State:
    """The targets of a batch and everything pending on them."""

    def __init__(
        self,
        targets: dict[Hashable, Target],
        seed: int | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        """Creates the state.

        Args:
            targets: The targets by name.
            seed: Seed of the call, drawn from torch's global generator if not given.
            shape: Spatial shape of the canvas, needed only when there are no raster targets.
        """

        self.targets = targets
        self.raster = {k: v for k, v in targets.items() if isinstance(v, RasterTarget)}
        self.geometry = {k: v for k, v in targets.items() if isinstance(v, GeometryTarget)}

        for name, target in targets.items():
            if not isinstance(target, Target):
                raise TypeError(f"Target {name!r} is a {type(target).__name__}, not a target.")

        self.seed = seed if seed is not None else int(torch.randint(0, 2**63 - 1, ()))

        self._init_raster(shape)
        self._init_geometry()

        # Pending work, in order: ("geometry", steps) and ("color", matrices by name, clamp).
        self._pending: list[list[Any]] = []

        # The geometric steps, in the runs they were resolved in, for replaying.
        self.history: list[list[Step]] = []

        # Draws cached per transform in this call, and what canvas transforms did.
        self._cache: dict[tuple[int, Hashable], Any] = {}
        self._alive: list[Any] = []
        self.records: dict[int, Any] = {}

    def _init_raster(self, shape: tuple[int, ...] | None) -> None:
        batch_sizes = [x.data.shape[0] for x in self.raster.values()]
        batch_sizes += [x.batch_size for x in self.geometry.values()]

        if not batch_sizes:
            raise ValueError("There is nothing to transform.")

        if len(set(batch_sizes)) != 1:
            raise ValueError(f"All targets must have the same batch size, not {batch_sizes}.")

        self.batch_size = batch_sizes[0]

        shapes = {tuple(x.data.shape[2:]) for x in self.raster.values()}
        if len(shapes) > 1:
            raise ValueError(
                f"All images and masks must have the same spatial shape, not {shapes}."
            )

        if shapes:
            spatial_shape = shapes.pop()
            if shape is not None and tuple(shape) != spatial_shape:
                raise ValueError(
                    f"The given shape {shape} is not that of the data, {spatial_shape}."
                )
        elif shape is not None:
            spatial_shape = tuple(int(x) for x in shape)
        else:
            raise ValueError("Without images or masks, pass the spatial `shape` of the canvas.")

        if len(spatial_shape) not in (2, 3):
            raise ValueError("Only 2D and 3D data is supported.")

        devices = {x.data.device for x in self.raster.values()}
        if len(devices) > 1:
            raise ValueError("All images and masks must be on the same device.")

        if devices:
            self.device = devices.pop()
        elif self.geometry:
            self.device = next(iter(self.geometry.values())).device
        else:
            self.device = torch.device("cpu")

        self.ndim = len(spatial_shape)
        self.input_shape = spatial_shape
        self.shape = spatial_shape

        # The shape of the canvas the raster data currently lives on.
        self._data_shape = spatial_shape

        self.data = {k: v.to_working() for k, v in self.raster.items()}
        self._owned = {k for k, v in self.raster.items() if self.data[k] is not v.data}

    def _init_geometry(self) -> None:
        ndim = self.ndim
        offset = -0.5 * size_of(self.input_shape)

        # The corners of the input canvas, always the first block.
        corners = torch.cartesian_prod(*(torch.tensor((-0.5, 0.5), dtype=torch.float64),) * ndim)
        corners = corners.view(-1, ndim) * size_of(self.input_shape)
        blocks = [corners.expand(self.batch_size, -1, -1)]
        valid = [torch.ones(self.batch_size, len(corners), dtype=torch.bool)]

        # Per target, where its block starts, how long it is, and its count per element.
        self._blocks: dict[Hashable, tuple[int, int, list[int]]] = {}
        start = len(corners)

        for name, target in self.geometry.items():
            items = target.encode(ndim)
            counts = [len(x) for x in items]
            length = max(counts, default=0)

            block = torch.zeros(self.batch_size, length, ndim, dtype=torch.float64)
            block_valid = torch.zeros(self.batch_size, length, dtype=torch.bool)
            for i, item in enumerate(items):
                block[i, : len(item)] = item + offset
                block_valid[i, : len(item)] = True

            blocks.append(block)
            valid.append(block_valid)

            self._blocks[name] = (start, length, counts)
            start += length

        self._points = torch.cat(blocks, dim=1)
        self._geometry_pending: list[Step] = []
        self.valid = torch.cat(valid, dim=1)

    def cached(self, transform: Any, key: Hashable, function: Callable[[], Any]) -> Any:
        """Returns what a transform drew for a key in this call, drawing it the first time."""

        cache_key = (id(transform), key)
        if cache_key not in self._cache:
            self._cache[cache_key] = function()

            # Keep the transform alive, so its id is not reused during the call.
            self._alive.append(transform)

        return self._cache[cache_key]

    def context(
        self, transform: Any, inverse: bool = False, indices: torch.Tensor | None = None
    ) -> Context:
        """Returns a fresh context for a transform, with its generators in their initial state."""

        return Context(self, transform, inverse, indices)

    def probability_mask(self, transform: Any, mask: torch.Tensor | None) -> torch.Tensor | None:
        """Combines a mask with a transform's own probability of being applied.

        Args:
            transform: The transform, with its probability `p`.
            mask: Which elements it is asked to apply to, or `None` for all.

        Returns:
            Which elements it applies to, of shape `(B,)`, or `None` for all.
        """

        p = transform.p
        if p >= 1:
            return mask

        def draw() -> torch.Tensor:
            generator = torch.Generator().manual_seed(mix_seed(self.seed, transform.seed, 3))

            return torch.rand(self.batch_size, dtype=torch.float64, generator=generator) < p

        applied = self.cached(transform, "probability", draw)

        return applied if mask is None else mask & applied

    def push(self, step: Step) -> None:
        """Adds a geometric step, and moves the geometry through it right away."""

        if step.input_shape != self.shape:
            raise RuntimeError(
                f"A step from {step.input_shape} does not fit the canvas {self.shape}."
            )

        if not self._pending or self._pending[-1][0] != "geometry":
            self._pending.append(["geometry", []])

        self._pending[-1][1].append(step)

        self._geometry_pending.append(step)
        self.shape = step.output_shape

    @property
    def points(self) -> torch.Tensor:
        """The points of all geometry, of shape `(B, P, n)`, moved through all steps so far.

        Points are moved lazily, only once something needs them, since moving them through a warp
        without a closed-form inverse takes iterations.
        """

        for step in self._geometry_pending:
            self._points = step.forward(self._points)

        self._geometry_pending.clear()

        return self._points

    def push_color(
        self, name: Hashable, matrices: torch.Tensor, clamp: bool, mask: torch.Tensor | None = None
    ) -> None:
        """Adds color matrices of shape `(B, C + 1, C + 1)` for an image.

        Args:
            name: Name of the image.
            matrices: The matrices, identity for the elements they do not apply to.
            clamp: Whether the result is clamped to `[0, 1]`.
            mask: Which elements they apply to, of shape `(B,)`, or `None` for all. Only those are
                clamped.
        """

        last = self._pending[-1] if self._pending else None
        if last is None or last[0] != "color" or last[2] != clamp:
            last = ["color", {}, clamp, {}]
            self._pending.append(last)

        previous = last[1].get(name)
        last[1][name] = matrices if previous is None else matrices @ previous

        # Which elements are clamped, where `None` is all of them.
        if name not in last[3]:
            last[3][name] = mask
        elif last[3][name] is not None:
            last[3][name] = None if mask is None else last[3][name] | mask

    def pending_color(self, name: Hashable) -> torch.Tensor | None:
        """Returns the color matrices pending for an image after all pending geometry, if any."""

        if self._pending and self._pending[-1][0] == "color":
            return self._pending[-1][1].get(name)

        return None

    def resolve(self, geometry_only: bool = False) -> None:
        """Applies everything pending to the raster data.

        Args:
            geometry_only: Whether to stop before trailing color matrices, which only happens when
                the geometry before them was all that needed resolving.
        """

        while self._pending:
            kind = self._pending[0][0]

            if geometry_only and kind == "color" and len(self._pending) == 1:
                return

            segment = self._pending.pop(0)

            if kind == "geometry":
                self._resolve_geometry(segment[1])
            else:
                self._resolve_color(segment[1], segment[2], segment[3])

    def _resolve_geometry(self, steps: list[Step]) -> None:
        self.history.append(steps)

        output_shape = steps[-1].output_shape
        sampling = Sampling(steps, self._data_shape, output_shape, self.batch_size)

        for name, target in self.raster.items():
            data, new = sampling.apply(target, self.data[name])

            self.data[name] = data
            if new:
                self._owned.add(name)
            else:
                self._owned.discard(name)

        self._data_shape = output_shape

    def _resolve_color(
        self,
        matrices: dict[Hashable, torch.Tensor],
        clamp: bool,
        clamped: dict[Hashable, torch.Tensor | None],
    ) -> None:
        for name, matrix in matrices.items():
            data = self.data[name]
            channels = data.shape[1]

            matrix = matrix.to(device=data.device, dtype=data.dtype)
            linear, offset = matrix[:, :channels, :channels], matrix[:, :channels, channels]

            flat = data.reshape(len(data), channels, -1)
            output = torch.baddbmm(offset[..., None], linear, flat).view(data.shape)

            if clamp:
                mask = clamped[name]
                if mask is None:
                    output = output.clamp_(0, 1)
                else:
                    mask = mask.to(device=data.device).view(-1, *(1,) * (data.ndim - 1))
                    output = torch.where(mask, output.clamp(0, 1), output)

            self.set(name, output)

    def set(self, name: Hashable, data: torch.Tensor) -> None:
        """Replaces the data of a raster target by a new tensor that is owned by the state."""

        self.data[name] = data
        self._owned.add(name)

    def writable(self, name: Hashable) -> torch.Tensor:
        """Returns the data of a raster target, copied first if it belongs to the caller."""

        if name not in self._owned:
            self.data[name] = self.data[name].clone(memory_format=torch.contiguous_format)
            self._owned.add(name)

        return self.data[name]

    def record(self, transform: Any, value: Any) -> None:
        """Records what a transform did, for its inverse."""

        self.records[id(transform)] = value
        self._alive.append(transform)

    def select(self, targets: tuple[Hashable, ...] | None) -> torch.Tensor:
        """Returns which points belong to geometry targets, see `Context.points`."""

        names = tuple(self._blocks) if targets is None else targets

        selected = torch.zeros_like(self.valid)
        for name in names:
            if name not in self._blocks:
                raise KeyError(f"There is no geometry target {name!r}.")

            start, length, _ = self._blocks[name]
            selected[:, start : start + length] = self.valid[:, start : start + length]

        # Elements without any selected point fall back to the corners of their content.
        empty = ~selected.any(dim=1)
        selected[:, : 2**self.ndim] |= empty[:, None]

        return selected

    def bounds(self, targets: tuple[Hashable, ...] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the lowest and highest point per element, see `Context.bounds`."""

        selected = self.select(targets)

        infinity = torch.tensor(float("inf"), dtype=torch.float64)
        points = self.points
        low = torch.where(selected[..., None], points, infinity).amin(dim=1)
        high = torch.where(selected[..., None], points, -infinity).amax(dim=1)

        return low, high

    def finish(self) -> dict[Hashable, Any]:
        """Resolves everything and returns the outputs by name, in the formats they came in."""

        self.resolve()

        outputs: dict[Hashable, Any] = {}
        all_points = self.points if self.geometry else None

        for name, target in self.targets.items():
            if isinstance(target, RasterTarget):
                outputs[name] = target.from_working(self.data[name])
            else:
                start, length, counts = self._blocks[name]
                points = all_points[:, start : start + length] + 0.5 * size_of(self.shape)
                items = [points[i, :count] for i, count in enumerate(counts)]
                outputs[name] = target.decode(items, self.shape)

        return outputs

    def replay(self) -> "Replay":
        """Returns a replay of the geometric steps of this call."""

        return Replay(list(self.history), self.input_shape, self.shape, self.batch_size)


class Replay:
    """The geometric transforms that one call applied, to apply again or to undo.

    A replay applies the same geometry to other data, such as labels that were not at hand when the
    batch was augmented, and undoes it, such as to bring predictions on augmented images back onto
    the original images for test-time augmentation. Photometric transforms are not replayed.
    """

    def __init__(
        self,
        segments: list[list[Step]],
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_size: int,
    ) -> None:
        """Creates the replay.

        Args:
            segments: The geometric steps, in the runs that were resolved in one resampling each.
            input_shape: Spatial shape of the canvas before them.
            output_shape: Spatial shape of the canvas after them.
            batch_size: Batch size B.
        """

        self.segments = segments
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size

    def apply(self, *inputs: Any, **targets: Any) -> Any:
        """Applies the geometry to data on the input canvas.

        The data is resampled exactly as the data of the call was, so a mask applied afterward is
        the same as one passed along with the call. Takes and returns data the way calling a
        transform does.
        """

        return self._run(self.segments, self.input_shape, inputs, targets)

    def inverse(self, *inputs: Any, **targets: Any) -> Any:
        """Undoes the geometry on data on the output canvas.

        All of it is undone in a single resampling. What lay outside the output canvas is filled
        by the padding of the targets. Takes and returns data the way calling a transform does.
        """

        steps = [x.inverted() for segment in reversed(self.segments) for x in reversed(segment)]

        return self._run([steps], self.output_shape, inputs, targets)

    def _run(
        self,
        segments: list[list[Step]],
        shape: tuple[int, ...],
        inputs: tuple[Any, ...],
        targets: dict[str, Any],
    ) -> Any:
        all_targets, pack = collect_targets(inputs, targets)
        state = State(all_targets, seed=0, shape=shape)

        if state.batch_size != self.batch_size:
            raise ValueError(
                f"The replay is of a batch of {self.batch_size}, not {state.batch_size}."
            )

        for segment in segments:
            for step in segment:
                state.push(step)

            state.resolve()

        return pack(state.finish())


def collect_targets(
    inputs: tuple[Any, ...], targets: dict[str, Any]
) -> tuple[dict[Hashable, Target], Callable[[dict[Hashable, Any]], Any]]:
    """Gathers the targets of a call, wrapping plain tensors as images.

    Args:
        inputs: Positional inputs.
        targets: Keyword inputs.

    Returns:
        The targets by name (positional ones by their index), and a function that packs the
        outputs the way the inputs came: one output for one positional input, a tuple for several,
        and a dictionary for keyword inputs.
    """

    if inputs and targets:
        raise ValueError("Pass targets either by position or by keyword, not both.")

    def wrap(value: Any) -> Target:
        if isinstance(value, Target):
            return value

        if isinstance(value, torch.Tensor):
            return Image(value)

        raise TypeError(f"Expected a tensor or a target, not a {type(value).__name__}.")

    if targets:
        wrapped: dict[Hashable, Target] = {k: wrap(v) for k, v in targets.items()}

        return wrapped, lambda outputs: outputs

    wrapped = {i: wrap(v) for i, v in enumerate(inputs)}

    if len(inputs) == 1:
        return wrapped, lambda outputs: outputs[0]

    return wrapped, lambda outputs: tuple(outputs[i] for i in range(len(inputs)))


def identity_where(mask: torch.Tensor | None, matrices: torch.Tensor) -> torch.Tensor:
    """Replaces the matrices of the elements outside a mask by identity matrices."""

    if mask is None:
        return matrices

    identity = mat.identity(len(matrices), matrices.shape[-1] - 1)

    return torch.where(mask[:, None, None], matrices, identity)
