"""Geometric transforms given by a matrix per element.

They are all fused into a single resampling with the other geometric transforms around them.
Angles are in degrees, and a positive angle turns the x axis toward the y axis, which is clockwise
on screen since y points down. Rotations, scales and shears are around the center of the canvas.
"""

import torch

from torchtransform import matrices as mat
from torchtransform.distributions import Parameter, symmetric, symmetric_log, to_distribution
from torchtransform.state import Context
from torchtransform.transform import MatrixTransform

# Axis names in coordinate order.
AXES: tuple[str, ...] = ("x", "y", "z")

# Parameter of a rotation axis: a fixed vector, "random" for a random axis per element, or `None`
# for the z axis, which turns within the xy plane.
AxisParameter = tuple[float, float, float] | str | None


def axis_index(axis: str, ndim: int) -> int:
    """Returns the coordinate index of an axis name."""

    if axis not in AXES[:ndim]:
        raise ValueError(f"Unknown axis {axis!r} for {ndim}D data, expected one of {AXES[:ndim]}.")

    return AXES.index(axis)


def rotation_matrices(context: Context, angles: torch.Tensor, axis: AxisParameter) -> torch.Tensor:
    """Returns rotation matrices for angles in degrees of shape `(B,)`, around an axis in 3D."""

    if context.ndim == 2:
        if axis is not None:
            raise ValueError("A rotation axis is only meaningful in 3D.")

        return mat.rotation_2d(angles)

    if axis is None:
        axes = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64).expand(len(angles), -1)
    elif axis == "random":
        axes = context.randn(len(angles), 3)
    elif isinstance(axis, str):
        axes = torch.zeros(len(angles), 3, dtype=torch.float64)
        axes[:, axis_index(axis, 3)] = 1
    else:
        axes = torch.tensor(axis, dtype=torch.float64).expand(len(angles), -1)

    return mat.rotation_3d(axes, angles)


class Rotate(MatrixTransform):
    def __init__(self, angle: Parameter, axis: AxisParameter = None, p: float = 1.0) -> None:
        """Rotates around the center of the canvas.

        Args:
            angle: Angle in degrees. A single number `a` is a fixed angle; use `(-a, a)` for a
                range. Positive angles turn the x axis toward the y axis, clockwise on screen.
            axis: The axis to rotate around in 3D, by the right-hand rule: `None` for the z axis
                (the same turn as in 2D), `"x"`, `"y"` or `"z"`, a vector, or `"random"` for a
                random axis per element.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.angle = to_distribution(angle)
        self.axis = axis

    def get_matrices(self, context: Context) -> torch.Tensor:
        return rotation_matrices(context, context.sample(self.angle), self.axis)

    def extra_repr(self) -> str:
        return f"angle={self.angle}, axis={self.axis}" + (f", p={self.p}" if self.p < 1 else "")


class RandomOrientation(MatrixTransform):
    def __init__(self, p: float = 1.0) -> None:
        """Rotates 3D data to an orientation drawn uniformly over all orientations.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

    def get_matrices(self, context: Context) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("RandomOrientation is only available in 3D.")

        return mat.rotation_from_quaternions(context.randn(context.batch_size, 4))


class Scale(MatrixTransform):
    def __init__(
        self,
        factor: Parameter = 1.0,
        x: Parameter = 1.0,
        y: Parameter = 1.0,
        z: Parameter = 1.0,
        p: float = 1.0,
    ) -> None:
        """Scales around the center of the canvas. Ranges are drawn uniformly in the logarithm.

        Args:
            factor: Factor for all axes. Above one enlarges the content.
            x: Additional factor along the x axis.
            y: Additional factor along the y axis.
            z: Additional factor along the z axis.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.factor = to_distribution(factor, log=True)
        self.factors = tuple(to_distribution(v, log=True) for v in (x, y, z))

    def get_matrices(self, context: Context) -> torch.Tensor:
        factors = context.sample(self.factor)[:, None]
        per_axis = torch.stack([context.sample(v) for v in self.factors[: context.ndim]], dim=1)

        return mat.scale(factors * per_axis)


class Translate(MatrixTransform):
    def __init__(
        self,
        x: Parameter = 0.0,
        y: Parameter = 0.0,
        z: Parameter = 0.0,
        relative: bool = False,
        p: float = 1.0,
    ) -> None:
        """Moves the content.

        Args:
            x: Distance along the x axis.
            y: Distance along the y axis.
            z: Distance along the z axis.
            relative: Whether distances are fractions of the size of the canvas, rather than pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.distances = tuple(to_distribution(v) for v in (x, y, z))
        self.relative = relative

    def get_matrices(self, context: Context) -> torch.Tensor:
        distances = torch.stack([context.sample(v) for v in self.distances[: context.ndim]], dim=1)

        if self.relative:
            distances = distances * context.size

        return mat.translation(distances)


class Shear(MatrixTransform):
    def __init__(
        self,
        xy: Parameter = 0.0,
        yx: Parameter = 0.0,
        xz: Parameter = 0.0,
        zx: Parameter = 0.0,
        yz: Parameter = 0.0,
        zy: Parameter = 0.0,
        p: float = 1.0,
    ) -> None:
        """Shears around the center of the canvas.

        Every argument is an angle in degrees: `xy` moves x along with y, as
        `x' = x + tan(xy) * y`, and so on.

        Args:
            xy: Shear of x along y.
            yx: Shear of y along x.
            xz: Shear of x along z (3D).
            zx: Shear of z along x (3D).
            yz: Shear of y along z (3D).
            zy: Shear of z along y (3D).
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.angles = {
            (0, 1): to_distribution(xy),
            (1, 0): to_distribution(yx),
            (0, 2): to_distribution(xz),
            (2, 0): to_distribution(zx),
            (1, 2): to_distribution(yz),
            (2, 1): to_distribution(zy),
        }

    def get_matrices(self, context: Context) -> torch.Tensor:
        ndim = context.ndim

        coefficients = torch.zeros(context.batch_size, ndim, ndim, dtype=torch.float64)
        for (i, j), angle in self.angles.items():
            if i < ndim and j < ndim:
                coefficients[:, i, j] = mat.degrees_to_shear(context.sample(angle))
            elif angle.low != 0 or angle.high != 0:
                raise ValueError("Shearing along z is only available in 3D.")

        return mat.shear(coefficients)


class Affine(MatrixTransform):
    def __init__(
        self,
        rotation: Parameter = 0.0,
        scale: Parameter = 1.0,
        aspect: Parameter = 1.0,
        translation: Parameter = 0.0,
        shear: Parameter = 0.0,
        axis: AxisParameter = None,
        p: float = 1.0,
    ) -> None:
        """Scales, shears, rotates and moves, in that order, as one transform.

        A single number for `rotation`, `translation` or `shear` is taken as a symmetric range
        around zero, and a single number for `scale` or `aspect` as a symmetric range around one,
        so `Affine(rotation=10, scale=1.2, translation=0.1)` is a typical random affine.

        Args:
            rotation: Angle in degrees, around `axis` in 3D.
            scale: Scale factor, drawn uniformly in the logarithm.
            aspect: Factor between the scales of the x and y axes (the x axis gets its square root,
                the y axis its inverse), drawn uniformly in the logarithm.
            translation: Distance, as a fraction of the size of the canvas, drawn per axis.
            shear: Shear angle in degrees, of x along y and of y along x, drawn for each.
            axis: The rotation axis in 3D, see `Rotate`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.rotation = to_distribution(symmetric(rotation))
        self.scale = to_distribution(symmetric_log(scale), log=True)
        self.aspect = to_distribution(symmetric_log(aspect), log=True)
        self.translation = to_distribution(symmetric(translation))
        self.shear = to_distribution(symmetric(shear))
        self.axis = axis

    def get_matrices(self, context: Context) -> torch.Tensor:
        batch_size, ndim = context.batch_size, context.ndim

        scales = context.sample(self.scale)[:, None].repeat(1, ndim)
        aspects = torch.sqrt(context.sample(self.aspect))
        scales[:, 0] *= aspects
        scales[:, 1] /= aspects

        coefficients = torch.zeros(batch_size, ndim, ndim, dtype=torch.float64)
        coefficients[:, 0, 1] = mat.degrees_to_shear(context.sample(self.shear))
        coefficients[:, 1, 0] = mat.degrees_to_shear(context.sample(self.shear))

        rotations = rotation_matrices(context, context.sample(self.rotation), self.axis)
        translations = mat.translation(context.sample(self.translation, ndim) * context.size)

        return translations @ rotations @ mat.shear(coefficients) @ mat.scale(scales)


class Flip(MatrixTransform):
    def __init__(self, axes: str | tuple[str, ...] = "x", p: float = 0.5) -> None:
        """Mirrors along axes. Exact: pixels are moved, not interpolated.

        Args:
            axes: The axis or axes to mirror: `"x"` mirrors left and right.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.axes = (axes,) if isinstance(axes, str) else tuple(axes)

    def get_matrices(self, context: Context) -> torch.Tensor:
        scales = torch.ones(context.batch_size, context.ndim, dtype=torch.float64)
        for axis in self.axes:
            scales[:, axis_index(axis, context.ndim)] = -1

        return mat.scale(scales)

    def extra_repr(self) -> str:
        return f"axes={self.axes}, p={self.p}"


class HorizontalFlip(Flip):
    def __init__(self, p: float = 0.5) -> None:
        """Mirrors left and right.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__("x", p)


class VerticalFlip(Flip):
    def __init__(self, p: float = 0.5) -> None:
        """Mirrors top and bottom.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__("y", p)


class QuarterTurn(MatrixTransform):
    def __init__(
        self,
        turns: Parameter = (0, 3),
        axes: tuple[str, str] = ("x", "y"),
        p: float = 1.0,
    ) -> None:
        """Turns by quarter turns, drawn per element. Exact: pixels are moved, not interpolated.

        Every element turns on its own, within the one canvas of the batch. On a square canvas
        nothing is lost; on another canvas, a turned element is cropped and padded to the canvas,
        so follow it with a crop or resize to the shape you want, or use a square canvas.

        Args:
            turns: Number of quarter turns from `axes[0]` toward `axes[1]` (clockwise on screen
                for x and y), by default any of the four, from zero to three.
            axes: The plane to turn in.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.turns = to_distribution(turns, integer=True)
        self.axes = axes

    def get_matrices(self, context: Context) -> torch.Tensor:
        first, second = (axis_index(x, context.ndim) for x in self.axes)

        radians = torch.round(context.sample(self.turns)) * (torch.pi / 2)
        cos, sin = torch.round(torch.cos(radians)), torch.round(torch.sin(radians))

        matrices = mat.identity(context.batch_size, context.ndim)
        matrices[:, first, first] = cos
        matrices[:, first, second] = -sin
        matrices[:, second, first] = sin
        matrices[:, second, second] = cos

        return matrices


class Transpose(MatrixTransform):
    def __init__(self, axes: tuple[str, str] = ("x", "y"), p: float = 0.5) -> None:
        """Swaps two axes. Exact: pixels are moved, not interpolated.

        Args:
            axes: The axes to swap.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.axes = axes

    def get_matrices(self, context: Context) -> torch.Tensor:
        first, second = (axis_index(x, context.ndim) for x in self.axes)

        matrices = mat.identity(context.batch_size, context.ndim)
        matrices[:, first, first] = 0
        matrices[:, second, second] = 0
        matrices[:, first, second] = 1
        matrices[:, second, first] = 1

        return matrices


class Symmetry(MatrixTransform):
    def __init__(
        self, axes: tuple[str, ...] | None = None, reflections: bool = True, p: float = 1.0
    ) -> None:
        """Applies a random symmetry of the square or cube, drawn uniformly per element.

        These are the permutations of the axes combined with mirroring any of them: the eight
        symmetries of a square in 2D, or the 48 of a cube in 3D. Exact: pixels are moved, not
        interpolated. On a canvas that is not square, see `QuarterTurn`.

        Args:
            axes: The axes to permute and mirror, by default all.
            reflections: Whether to include mirror images, rather than only rotations.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.axes = axes
        self.reflections = reflections

    def get_matrices(self, context: Context) -> torch.Tensor:
        batch_size, ndim = context.batch_size, context.ndim

        axes = self.axes if self.axes is not None else AXES[:ndim]
        indices = torch.tensor([axis_index(x, ndim) for x in axes])
        count = len(indices)

        # A random permutation per element, by sorting random values, with random signs.
        permutations = torch.argsort(context.rand(batch_size, count), dim=1)
        linear = torch.nn.functional.one_hot(permutations, count).to(dtype=torch.float64)
        linear = linear * (2 * context.randint(0, 2, batch_size, count, 1) - 1)

        # Without reflections, a matrix with determinant -1 gets its first row negated.
        if not self.reflections:
            linear[:, 0] *= torch.linalg.det(linear)[:, None]

        matrices = mat.identity(batch_size, ndim)
        matrices[:, indices[:, None], indices[None, :]] = linear

        return matrices


class Perspective(MatrixTransform):
    def __init__(self, distortion: Parameter = (0.0, 0.3), p: float = 1.0) -> None:
        """Changes the perspective, by moving the four corners of a 2D canvas.

        Args:
            distortion: How far every corner moves along each axis, at most, as a fraction of half
                the size of the canvas. Every corner moves by a uniform amount in either direction.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.distortion = to_distribution(distortion)

    def get_matrices(self, context: Context) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError("Perspective is only available in 2D.")

        batch_size = context.batch_size
        half = 0.5 * context.size

        corners = torch.tensor(((-1, -1), (1, -1), (1, 1), (-1, 1)), dtype=torch.float64) * half
        corners = corners.expand(batch_size, -1, -1)

        distortions = context.sample(self.distortion)[:, None, None]
        moves = (2 * context.rand(batch_size, 4, 2) - 1) * distortions * half

        return mat.homography(corners, corners + moves)


class AffineWithinBounds(MatrixTransform):
    def __init__(
        self,
        rotation: Parameter = (-5.0, 5.0),
        scale: Parameter = (0.8, 1.2),
        aspect: Parameter = 1.0,
        targets: tuple[str, ...] | None = None,
        p: float = 1.0,
    ) -> None:
        """Rotates, scales and moves at random, but keeps the geometry on the canvas.

        The rotation is drawn freely. The scale is then limited so that the rotated geometry fits
        the canvas, and the translation drawn from where it still fits. Elements without geometry
        keep their whole content on the canvas instead. In 3D, the rotation is around the z axis.

        Args:
            rotation: Angle in degrees.
            scale: Scale factor, drawn uniformly in the logarithm, before it is limited.
            aspect: Factor between the scales of the x and y axes, drawn uniformly in the
                logarithm (the x axis gets its square root, the y axis its inverse).
            targets: The names of the geometry targets to keep on the canvas, by default all.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.rotation = to_distribution(rotation)
        self.scale = to_distribution(scale, log=True)
        self.aspect = to_distribution(symmetric_log(aspect), log=True)
        self.targets = targets

    def get_matrices(self, context: Context) -> torch.Tensor:
        batch_size, ndim = context.batch_size, context.ndim

        points, selected = context.points(self.targets)

        rotations = rotation_matrices(context, context.sample(self.rotation), None)
        rotated = mat.apply(rotations, points)

        infinity = torch.tensor(float("inf"), dtype=torch.float64)
        low = torch.where(selected[..., None], rotated, infinity).amin(dim=1)
        high = torch.where(selected[..., None], rotated, -infinity).amax(dim=1)
        extent = high - low

        aspects = torch.sqrt(context.sample(self.aspect))
        ratios = torch.ones(batch_size, ndim, dtype=torch.float64)
        ratios[:, 0] = aspects
        ratios[:, 1] = 1 / aspects

        # The largest scale at which the rotated geometry still fits. Geometry without extent along
        # an axis does not limit the scale.
        limits = torch.where(
            extent > 0, context.size / (extent * ratios).clamp(min=1e-12), infinity
        )
        limit = limits.amin(dim=1)

        highest = torch.minimum(torch.full_like(limit, self.scale.high), limit)
        lowest = torch.minimum(torch.full_like(limit, self.scale.low), highest)
        fraction = context.rand(batch_size)
        scales = torch.exp(torch.log(lowest) + fraction * (torch.log(highest) - torch.log(lowest)))
        scales = scales[:, None] * ratios

        # Where the scaled geometry can go and still be on the canvas.
        half = 0.5 * context.size
        low_translation = -half - scales * low
        high_translation = half - scales * high
        translations = low_translation + context.rand(batch_size, ndim) * (
            high_translation - low_translation
        )

        return mat.translation(translations) @ mat.scale(scales) @ rotations
