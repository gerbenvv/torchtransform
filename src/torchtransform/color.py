"""Color transforms that are affine per pixel, fused into one matrix per element.

A run of consecutive color transforms costs a single pass over the pixels: their matrices are
multiplied first, and the values clamped to `[0, 1]` once at the end. Images are expected in
`[0, 1]`. Transforms that mix channels need RGB images.
"""

import math

import torch

from torchtransform.distributions import Parameter, symmetric, to_distribution
from torchtransform.functional import LUMA_WEIGHTS
from torchtransform.state import ColorView, Context
from torchtransform.transform import ColorTransform


def affine(linear: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
    """Returns color matrices of shape `(B, C + 1, C + 1)` from linear parts and offsets."""

    batch_size, channels, _ = linear.shape

    matrices = torch.zeros(batch_size, channels + 1, channels + 1, dtype=torch.float64)
    matrices[:, :channels, :channels] = linear
    matrices[:, channels, channels] = 1

    if offset is not None:
        matrices[:, :channels, channels] = offset

    return matrices


def eye(batch_size: int, channels: int) -> torch.Tensor:
    """Returns identity linear parts of shape `(B, C, C)`."""

    return torch.eye(channels, dtype=torch.float64).expand(batch_size, -1, -1).clone()


def luma(channels: int) -> torch.Tensor:
    """Returns the weights of the channels in the luma, of shape `(C,)`.

    These are the BT.601 weights for RGB, and equal weights otherwise.
    """

    if channels == 3:
        return torch.tensor(LUMA_WEIGHTS, dtype=torch.float64)

    return torch.full((channels,), 1 / channels, dtype=torch.float64)


def require_rgb(transform: ColorTransform, image: ColorView) -> None:
    """Raises unless an image has three channels."""

    if image.channels != 3:
        raise ValueError(
            f"{type(transform).__name__} needs RGB images, not {image.channels} channels."
        )


def brightness_matrices(factors: torch.Tensor, channels: int) -> torch.Tensor:
    """Returns matrices that multiply colors by factors of shape `(B,)` or `(B, C)`."""

    if factors.ndim == 1:
        factors = factors[:, None].expand(-1, channels)

    return affine(torch.diag_embed(factors))


def contrast_matrices(factors: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Returns matrices that blend colors with their mean luma by factors of shape `(B,)`."""

    batch_size, channels = means.shape
    gray = (means * luma(channels)).sum(dim=1)

    linear = eye(batch_size, channels) * factors[:, None, None]
    offset = ((1 - factors) * gray)[:, None].expand(-1, channels)

    return affine(linear, offset)


def saturation_matrices(factors: torch.Tensor) -> torch.Tensor:
    """Returns matrices that blend RGB colors with their luma by factors of shape `(B,)`."""

    batch_size = len(factors)
    gray = luma(3)[None, None].expand(batch_size, 3, 3)

    linear = eye(batch_size, 3) * factors[:, None, None] + (1 - factors)[:, None, None] * gray

    return affine(linear)


def hue_matrices(shifts: torch.Tensor) -> torch.Tensor:
    """Returns matrices that turn the hue of RGB colors by fractions of a turn, of shape `(B,)`.

    The turn is around the gray axis, weighted to keep the luma, as the `hue-rotate` filter of CSS.
    """

    radians = 2 * math.pi * shifts
    cos, sin = torch.cos(radians), torch.sin(radians)

    base = torch.tensor(((0.213, 0.715, 0.072),) * 3, dtype=torch.float64)
    cos_part = torch.tensor(
        ((0.787, -0.715, -0.072), (-0.213, 0.285, -0.072), (-0.213, -0.715, 0.928)),
        dtype=torch.float64,
    )
    sin_part = torch.tensor(
        ((-0.213, -0.715, 0.928), (0.143, 0.140, -0.283), (-0.787, 0.715, 0.072)),
        dtype=torch.float64,
    )

    linear = base + cos[:, None, None] * cos_part + sin[:, None, None] * sin_part

    return affine(linear)


class Brightness(ColorTransform):
    def __init__(
        self, factor: Parameter = (0.8, 1.2), per_channel: bool = False, p: float = 1.0
    ) -> None:
        """Multiplies the values.

        Args:
            factor: Factor: zero is black, one leaves the image alone.
            per_channel: Whether every channel draws its own factor, which also shifts the color
                balance.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.factor = to_distribution(factor)
        self.per_channel = per_channel

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        factors = (
            context.sample(self.factor, image.channels)
            if self.per_channel
            else context.sample(self.factor)
        )

        return brightness_matrices(factors, image.channels)


class Contrast(ColorTransform):
    def __init__(self, factor: Parameter = (0.8, 1.2), p: float = 1.0) -> None:
        """Blends with the mean gray of the image.

        Args:
            factor: Factor: zero is uniformly gray, one leaves the image alone, above one raises the
                contrast.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.factor = to_distribution(factor)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        return contrast_matrices(context.sample(self.factor), image.mean())


class Saturation(ColorTransform):
    def __init__(self, factor: Parameter = (0.5, 1.5), p: float = 1.0) -> None:
        """Blends RGB images with their gray version.

        Args:
            factor: Factor: zero is gray, one leaves the image alone, above one saturates.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.factor = to_distribution(factor)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        require_rgb(self, image)

        return saturation_matrices(context.sample(self.factor))


class Hue(ColorTransform):
    def __init__(self, shift: Parameter = (-0.1, 0.1), p: float = 1.0) -> None:
        """Turns the hue of RGB images.

        The turn is around the gray axis, weighted to keep the luma, as the `hue-rotate` filter of
        CSS does, which makes it a matrix that fuses with the other color transforms.

        Args:
            shift: Fraction of a full turn, in `[-0.5, 0.5]`. A single number `a` is the range
                `(-a, a)`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.shift = to_distribution(symmetric(shift))

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        require_rgb(self, image)

        return hue_matrices(context.sample(self.shift))


class ColorJitter(ColorTransform):
    def __init__(
        self,
        brightness: Parameter = 0.0,
        contrast: Parameter = 0.0,
        saturation: Parameter = 0.0,
        hue: Parameter = 0.0,
        p: float = 1.0,
    ) -> None:
        """Changes brightness, contrast, saturation and hue in a random order per element.

        As in torchvision, a single number `a` is the range `(1 - a, 1 + a)` for the first three
        and `(-a, a)` for the hue. The four are one matrix, so this is a single pass.

        Args:
            brightness: Brightness factor, see `Brightness`.
            contrast: Contrast factor, see `Contrast`.
            saturation: Saturation factor, see `Saturation`.
            hue: Hue shift, see `Hue`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        def around_one(value: Parameter) -> Parameter:
            if isinstance(value, (int, float)):
                return (max(0.0, 1 - value), 1 + value)

            return value

        self.brightness = to_distribution(around_one(brightness))
        self.contrast = to_distribution(around_one(contrast))
        self.saturation = to_distribution(around_one(saturation))
        self.hue = to_distribution(symmetric(hue))

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        batch_size, channels = context.batch_size, image.channels

        brightness = context.sample(self.brightness)
        contrast = context.sample(self.contrast)
        saturation = context.sample(self.saturation)
        hue = context.sample(self.hue)
        orders = torch.argsort(context.rand(batch_size, 4), dim=1)

        uses_color = (self.saturation.low, self.saturation.high, self.hue.low, self.hue.high) != (
            1.0,
            1.0,
            0.0,
            0.0,
        )
        if uses_color and channels != 3:
            raise ValueError(
                f"ColorJitter needs RGB images for saturation and hue, not {channels} channels."
            )

        identity = affine(eye(batch_size, channels))
        candidates = torch.stack(
            (
                brightness_matrices(brightness, channels),
                affine(eye(batch_size, channels) * contrast[:, None, None]),
                saturation_matrices(saturation) if channels == 3 else identity,
                hue_matrices(hue) if channels == 3 else identity,
            ),
            dim=1,
        )

        means = image.mean()
        homogeneous = torch.cat((means, torch.ones(batch_size, 1, dtype=torch.float64)), dim=1)[
            ..., None
        ]
        weights = luma(channels)
        elements = torch.arange(batch_size)

        matrices = identity
        for position in range(4):
            steps = orders[:, position]
            chosen = candidates[elements, steps]

            # The contrast step blends with the mean luma of the image as it is at that point.
            contrasted = steps == 1
            if bool(contrasted.any()):
                gray = ((matrices @ homogeneous)[:, :channels, 0] * weights).sum(dim=1)
                offsets = ((1 - contrast) * gray)[:, None].expand(-1, channels)
                chosen[:, :channels, channels] = torch.where(
                    contrasted[:, None], offsets, chosen[:, :channels, channels]
                )

            matrices = chosen @ matrices

        return matrices


class Grayscale(ColorTransform):
    def __init__(self, p: float = 1.0) -> None:
        """Turns RGB images gray, keeping their three channels.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        require_rgb(self, image)

        return affine(luma(3)[None, None].expand(context.batch_size, 3, 3).clone())


class Invert(ColorTransform):
    def __init__(self, p: float = 1.0) -> None:
        """Inverts the values, `x` becoming `1 - x`.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        batch_size, channels = context.batch_size, image.channels

        return affine(
            -eye(batch_size, channels), torch.ones(batch_size, channels, dtype=torch.float64)
        )


class RGBShift(ColorTransform):
    def __init__(self, shift: Parameter = (-0.1, 0.1), p: float = 1.0) -> None:
        """Adds a value to every channel, drawn per channel.

        Args:
            shift: Value to add. A single number `a` is the range `(-a, a)`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.shift = to_distribution(symmetric(shift))

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        batch_size, channels = context.batch_size, image.channels

        return affine(eye(batch_size, channels), context.sample(self.shift, channels))


class ChannelShuffle(ColorTransform):
    def __init__(self, p: float = 1.0) -> None:
        """Permutes the channels at random.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        batch_size, channels = context.batch_size, image.channels
        permutations = torch.argsort(context.rand(batch_size, channels), dim=1)

        return affine(torch.nn.functional.one_hot(permutations, channels).to(dtype=torch.float64))


class ChannelDropout(ColorTransform):
    def __init__(self, fill: float = 0.0, p: float = 1.0) -> None:
        """Replaces one channel, drawn at random, by a value.

        Args:
            fill: The value.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.fill = float(fill)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        batch_size, channels = context.batch_size, image.channels
        dropped = context.randint(0, channels, batch_size)

        keep = 1 - torch.nn.functional.one_hot(dropped, channels).to(dtype=torch.float64)

        return affine(torch.diag_embed(keep), (1 - keep) * self.fill)


class Sepia(ColorTransform):
    def __init__(self, strength: Parameter = 1.0, p: float = 1.0) -> None:
        """Tones RGB images sepia.

        Args:
            strength: How far: zero leaves the image alone, one is fully sepia.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.strength = to_distribution(strength)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        require_rgb(self, image)

        sepia = torch.tensor(
            ((0.393, 0.769, 0.189), (0.349, 0.686, 0.168), (0.272, 0.534, 0.131)),
            dtype=torch.float64,
        )
        strengths = context.sample(self.strength)[:, None, None]

        return affine(eye(context.batch_size, 3) * (1 - strengths) + sepia * strengths)


class AutoContrast(ColorTransform):
    def __init__(self, p: float = 1.0) -> None:
        """Stretches every channel to span `[0, 1]`.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        data = image.data()
        dims = tuple(range(2, data.ndim))

        low = data.amin(dim=dims).to(dtype=torch.float64).cpu()
        high = data.amax(dim=dims).to(dtype=torch.float64).cpu()

        # A flat channel is left alone rather than blown up.
        scale = torch.where(
            high - low > 1e-6, 1 / (high - low).clamp(min=1e-6), torch.ones_like(low)
        )
        offset = torch.where(high - low > 1e-6, -low * scale, torch.zeros_like(low))

        return affine(torch.diag_embed(scale), offset)


class Normalize(ColorTransform):
    # The result is meant to leave [0, 1].
    clamp: bool = False

    def __init__(self, mean: float | tuple[float, ...], std: float | tuple[float, ...]) -> None:
        """Normalizes every channel, `x` becoming `(x - mean) / std`. Not clamped.

        Put it last. Its inverse, with `Inverse` or on a `Replay`, undoes it.

        Args:
            mean: Mean, one for all channels or one per channel.
            std: Standard deviation, one for all channels or one per channel.
        """

        super().__init__()

        self.mean = torch.as_tensor(mean, dtype=torch.float64).flatten()
        self.std = torch.as_tensor(std, dtype=torch.float64).flatten()

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        batch_size, channels = context.batch_size, image.channels

        mean = self.mean.expand(channels)
        std = self.std.expand(channels)

        linear = torch.diag_embed((1 / std).expand(batch_size, -1))

        return affine(linear, (-mean / std).expand(batch_size, -1))

    def extra_repr(self) -> str:
        return f"mean={self.mean.tolist()}, std={self.std.tolist()}"


class ColorMatrix(ColorTransform):
    def __init__(self, matrix: torch.Tensor, clamp: bool = True, p: float = 1.0) -> None:
        """Applies a fixed color matrix.

        Args:
            matrix: A linear matrix of shape `(C, C)`, or an affine one of shape `(C, C + 1)` or
                `(C + 1, C + 1)`.
            clamp: Whether to clamp the result to `[0, 1]`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.matrix = torch.as_tensor(matrix, dtype=torch.float64)
        self.clamp = clamp

    def get_color_matrices(self, context: Context, image: ColorView) -> torch.Tensor:
        channels = image.channels
        matrix = self.matrix

        if matrix.shape[0] != channels:
            raise ValueError(f"The color matrix is for {matrix.shape[0]} channels, not {channels}.")

        full = torch.eye(channels + 1, dtype=torch.float64)
        full[:channels, : matrix.shape[1]] = matrix[:channels]

        return full.expand(context.batch_size, -1, -1).clone()
