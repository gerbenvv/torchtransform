"""Smooth nonlinear geometric transforms.

Warps are fused into the single resampling of the geometric transforms around them. They are
evaluated on a grid every `grid_step` pixels and interpolated in between, since they are smooth,
which makes them cheap even on large images.
"""

import math

import torch

from torchtransform.distributions import Parameter, symmetric, to_distribution
from torchtransform.state import Context
from torchtransform.transform import WarpTransform

# Most control values a B-spline evaluation gathers at once, per batch element.
BSPLINE_CHUNK_SIZE: int = 1 << 22


def expand(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Reshapes a per-element parameter to broadcast against coordinates.

    Args:
        value: A scalar per element, of shape `(B,)`, or a vector per element, of shape `(B, k)`.
        coordinates: Coordinates of shape `(B, ..., n)`.

    Returns:
        The parameter, of shape `(B, 1, ..., 1)` or `(B, 1, ..., 1, k)`, broadcasting against the
        coordinates without their last dimension, or with it, respectively.
    """

    return value.reshape(value.shape[0], *(1,) * (coordinates.ndim - 2), *value.shape[1:])


def cubic_bspline_weights(fractions: torch.Tensor) -> torch.Tensor:
    """Returns the four uniform cubic B-spline weights for fractions in `[0, 1)`, stacked last."""

    u = fractions
    u2 = u * u
    u3 = u2 * u

    return torch.stack(
        ((1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6),
        dim=-1,
    )


def evaluate_bspline(control: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Evaluates a uniform cubic B-spline of control values at positions.

    Args:
        control: Control values of shape `(B, K, *G)` on a grid `G` in tensor order.
        positions: Positions of shape `(B, ..., n)` in grid units in `(x, y[, z])` order, where
            control point `k` is at position `k`. The spline is defined from position one to two
            before the last control point, and positions are clamped to that.

    Returns:
        The values, of shape `(B, ..., K)`.
    """

    batch_size, channels = control.shape[:2]
    grid_shape = control.shape[2:]
    ndim = len(grid_shape)

    flat_positions = positions.reshape(batch_size, -1, ndim)
    count = flat_positions.shape[1]

    # Per axis, in tensor order: the first of the four control points and the weights.
    starts = []
    weights = []
    for axis in range(ndim):
        size = grid_shape[axis]
        position = flat_positions[..., ndim - 1 - axis].clamp(1, size - 2 - 1e-6)

        # The interval from control point `j` to `j + 1` uses the points `j - 1` to `j + 2`.
        interval = torch.floor(position)
        starts.append(interval.to(dtype=torch.int64) - 1)
        weights.append(cubic_bspline_weights(position - interval))

    strides = torch.tensor(
        [math.prod(grid_shape[axis + 1 :]) for axis in range(ndim)], device=control.device
    )
    flat_control = control.reshape(batch_size, channels, -1)

    # All 4 ** n control points of every position at once: their offsets from the first one, and
    # their weights, the products of the weights along every axis.
    offsets = torch.cartesian_prod(*(torch.arange(4, device=control.device),) * ndim).view(-1, ndim)
    base = sum(starts[a] * strides[a] for a in range(ndim))
    index_offsets = (offsets * strides).sum(dim=1)

    weight = weights[0][..., offsets[:, 0]]
    for axis in range(1, ndim):
        weight = weight * weights[axis][..., offsets[:, axis]]

    # In chunks of positions, so the gathered control values stay a modest size.
    chunk = max(1, BSPLINE_CHUNK_SIZE // (len(offsets) * channels))
    values = []
    for first in range(0, count, chunk):
        index = base[:, first : first + chunk, None] + index_offsets
        length = index.shape[1]

        gathered = torch.gather(
            flat_control, 2, index.reshape(batch_size, 1, -1).expand(-1, channels, -1)
        )
        gathered = gathered.view(batch_size, channels, length, len(offsets))

        values.append((gathered * weight[:, None, first : first + chunk]).sum(dim=-1))

    values = torch.cat(values, dim=2)

    return values.transpose(1, 2).reshape(*positions.shape[:-1], channels)


class ElasticWarp(WarpTransform):
    def __init__(
        self,
        magnitude: Parameter = (0.0, 8.0),
        spacing: float = 32.0,
        grid_step: float = 4.0,
        p: float = 1.0,
    ) -> None:
        """Deforms elastically, with a smooth random displacement drawn per element.

        The displacement is a cubic B-spline through random displacements on a coarse grid of
        control points, so it is smooth everywhere and costs little to draw.

        Args:
            magnitude: Standard deviation of the displacement of the control points, in pixels.
            spacing: Distance between control points, in pixels, roughly the size of a bump.
            grid_step: Spacing of the grid the warp is evaluated on, in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(grid_step, p)

        self.magnitude = to_distribution(magnitude)
        self.spacing = float(spacing)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        cells = [max(1, round(s / self.spacing)) for s in context.shape]
        magnitudes = context.sample(self.magnitude)

        control = context.randn(context.batch_size, context.ndim, *(c + 3 for c in cells))
        control = control * magnitudes.view(-1, *(1,) * (context.ndim + 1))

        cell_size = context.size / torch.tensor(cells[::-1], dtype=torch.float64)

        return dict(control=control, cell_size=cell_size.expand(context.batch_size, -1).clone())

    def backward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        cell_size = expand(parameters["cell_size"], coordinates)
        grid_shape = parameters["control"].shape[2:]
        half = cell_size * torch.tensor(
            [(g - 3) / 2 for g in grid_shape[::-1]],
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        # Control point `k` lies `k - 1` cells from the start of the canvas, so the canvas spans
        # the positions from one to two before the last control point.
        positions = (coordinates + half) / cell_size + 1

        return coordinates + evaluate_bspline(parameters["control"], positions)

    def extra_repr(self) -> str:
        return f"magnitude={self.magnitude}, spacing={self.spacing}, " + super().extra_repr()


class LensDistortion(WarpTransform):
    def __init__(self, k: Parameter = (-0.2, 0.2), grid_step: float = 4.0, p: float = 1.0) -> None:
        """Distorts radially, as a lens does.

        With `r` the distance from the center relative to half the diagonal of the canvas, a pixel
        shows what lies at `r * (1 + k * r ** 2)`. Positive `k` gives barrel distortion, negative
        `k` pincushion distortion.

        Args:
            k: The distortion coefficient. A single number `a` is the range `(-a, a)`.
            grid_step: Spacing of the grid the warp is evaluated on, in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(grid_step, p)

        self.k = to_distribution(symmetric(k))

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        radius = 0.5 * torch.linalg.norm(context.size)

        return dict(k=context.sample(self.k) / radius**2)

    def backward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        k = expand(parameters["k"], coordinates)[..., None]
        squared = (coordinates * coordinates).sum(dim=-1, keepdim=True)

        return coordinates * (1 + k * squared)

    def forward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        k = expand(parameters["k"], coordinates)[..., None]
        target = torch.linalg.norm(coordinates, dim=-1, keepdim=True)

        # Newton's method for the radius `r` with `r * (1 + k * r ** 2) = target`.
        radius = target.clone()
        for _ in range(20):
            value = radius * (1 + k * radius * radius) - target
            slope = (1 + 3 * k * radius * radius).clamp(min=1e-3)
            radius = (radius - value / slope).clamp(min=0)

        return coordinates * torch.where(
            target > 0, radius / target.clamp(min=1e-12), torch.ones_like(target)
        )


class Twirl(WarpTransform):
    def __init__(
        self,
        angle: Parameter = 30.0,
        radius: Parameter = (0.2, 0.5),
        center: Parameter = (-0.25, 0.25),
        grid_step: float = 4.0,
        p: float = 1.0,
    ) -> None:
        """Twirls around a point, most at the point and fading out to nothing at a radius.

        In 3D it twirls around an axis parallel to the z axis.

        Args:
            angle: Angle in degrees at the center. A single number `a` is the range `(-a, a)`.
            radius: Radius, as a fraction of the shortest side of the canvas.
            center: Offset of the center from the center of the canvas along the x and y axes, as
                fractions of the size of the canvas.
            grid_step: Spacing of the grid the warp is evaluated on, in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(grid_step, p)

        self.angle = to_distribution(symmetric(angle))
        self.radius = to_distribution(radius)
        self.center = to_distribution(center)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        shortest = float(context.size[:2].min())

        return dict(
            angle=torch.deg2rad(context.sample(self.angle)),
            radius=context.sample(self.radius) * shortest,
            center=context.sample(self.center, 2) * context.size[:2],
        )

    def _turn(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor], sign: float
    ) -> torch.Tensor:
        center = expand(parameters["center"], coordinates)
        radius = expand(parameters["radius"], coordinates)[..., None]
        angle = expand(parameters["angle"], coordinates)[..., None]

        offsets = coordinates[..., :2] - center
        distances = torch.linalg.norm(offsets, dim=-1, keepdim=True)

        # The turn fades out smoothly: its value and slope reach zero at the radius. The distance
        # to the center stays the same, so the inverse turns back by the same amount.
        falloff = (1 - distances / radius).clamp(min=0) ** 2
        radians = sign * angle * falloff
        cos, sin = torch.cos(radians), torch.sin(radians)

        x, y = offsets[..., :1], offsets[..., 1:]
        turned = torch.cat((cos * x - sin * y, sin * x + cos * y), dim=-1) + center

        return torch.cat((turned, coordinates[..., 2:]), dim=-1)

    def backward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._turn(coordinates, parameters, -1.0)

    def forward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._turn(coordinates, parameters, 1.0)


class Wave(WarpTransform):
    def __init__(
        self,
        amplitude: Parameter = (0.0, 4.0),
        wavelength: Parameter = (32.0, 128.0),
        angle: Parameter = (0.0, 180.0),
        grid_step: float = 4.0,
        p: float = 1.0,
    ) -> None:
        """Displaces along a sine wave, as on a rippled or wavy page.

        Content is moved across the direction of the wave, by an amount that follows a sine along
        it. In 3D, the wave lies in the xy plane.

        Args:
            amplitude: Amplitude in pixels.
            wavelength: Wavelength in pixels.
            angle: Direction of the wave in degrees, from the x axis toward the y axis.
            grid_step: Spacing of the grid the warp is evaluated on, in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(grid_step, p)

        self.amplitude = to_distribution(amplitude)
        self.wavelength = to_distribution(wavelength)
        self.angle = to_distribution(angle)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        radians = torch.deg2rad(context.sample(self.angle))

        return dict(
            amplitude=context.sample(self.amplitude),
            frequency=2 * math.pi / context.sample(self.wavelength),
            direction=torch.stack((torch.cos(radians), torch.sin(radians)), dim=1),
            phase=2 * math.pi * context.rand(context.batch_size),
        )

    def backward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        direction = expand(parameters["direction"], coordinates)
        amplitude = expand(parameters["amplitude"], coordinates)[..., None]
        frequency = expand(parameters["frequency"], coordinates)[..., None]
        phase = expand(parameters["phase"], coordinates)[..., None]

        along = (coordinates[..., :2] * direction).sum(dim=-1, keepdim=True)
        across = torch.cat((-direction[..., 1:], direction[..., :1]), dim=-1)

        moved = coordinates[..., :2] + amplitude * torch.sin(frequency * along + phase) * across

        return torch.cat((moved, coordinates[..., 2:]), dim=-1)


class GridDistortion(WarpTransform):
    def __init__(
        self,
        steps: int = 5,
        distortion: Parameter = 0.3,
        grid_step: float = 4.0,
        p: float = 1.0,
    ) -> None:
        """Distorts along every axis on its own, by stretching and squeezing the cells of a grid.

        The canvas is divided into `steps` cells along every axis, and every cell is stretched or
        squeezed by its own factor, so straight lines along the axes stay straight. Both directions
        of the map are piecewise linear, so geometry moves exactly.

        Args:
            steps: Number of cells along every axis.
            distortion: How much a cell changes in size, as a fraction: a cell of size `s` becomes
                one of size `s * (1 + d)`. A single number `a` is the range `(-a, a)`; keep it
                within `(-1, 1)`.
            grid_step: Spacing of the grid the warp is evaluated on, in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(grid_step, p)

        if steps < 1:
            raise ValueError("A grid distortion needs at least one step.")

        self.steps = int(steps)
        self.distortion = to_distribution(symmetric(distortion))

        if self.distortion.low <= -1:
            raise ValueError("A distortion of -1 or less would fold cells over.")

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        batch_size, ndim = context.batch_size, context.ndim

        # The positions of the cell boundaries after the distortion, per axis, from the start to
        # the end of the canvas.
        sizes = 1 + context.sample(self.distortion, ndim, self.steps)
        boundaries = torch.cat(
            (torch.zeros(batch_size, ndim, 1, dtype=torch.float64), sizes.cumsum(dim=2)), dim=2
        )
        boundaries = boundaries / boundaries[..., -1:]

        size = context.size.view(1, ndim, 1)

        return dict(
            boundaries=(boundaries - 0.5) * size, size=context.size.expand(batch_size, -1).clone()
        )

    def _map(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor], forward: bool
    ) -> torch.Tensor:
        batch_size, ndim = coordinates.shape[0], coordinates.shape[-1]
        flat = coordinates.reshape(batch_size, -1, ndim)

        distorted = parameters["boundaries"]
        size = parameters["size"]
        steps = distorted.shape[-1] - 1

        regular = (
            torch.linspace(0, 1, steps + 1, device=coordinates.device, dtype=coordinates.dtype)
            - 0.5
        )
        regular = regular.view(1, 1, -1) * size[..., None]

        # The backward map takes the regular boundaries to the distorted ones, since a pixel at a
        # regular boundary shows what lies at the distorted one; the forward map the other way.
        sources, targets = (distorted, regular) if forward else (regular, distorted)

        mapped = []
        for axis in range(ndim):
            values = flat[..., axis].contiguous()
            knots = sources[:, axis].contiguous()

            cell = (torch.searchsorted(knots, values) - 1).clamp(0, steps - 1)
            low, high = torch.gather(knots, 1, cell), torch.gather(knots, 1, cell + 1)
            fraction = (values - low) / (high - low).clamp(min=1e-12)

            target_low = torch.gather(targets[:, axis], 1, cell)
            target_high = torch.gather(targets[:, axis], 1, cell + 1)
            mapped.append(target_low + fraction * (target_high - target_low))

        return torch.stack(mapped, dim=-1).view(coordinates.shape)

    def backward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._map(coordinates, parameters, forward=False)

    def forward_coordinates(
        self, coordinates: torch.Tensor, parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self._map(coordinates, parameters, forward=True)
