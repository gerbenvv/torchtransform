"""Photometric transforms that change pixels: tone curves, filters, noise and artifacts.

Every parameter is drawn per batch element, and every transform works on the whole batch at once.
Images are expected in `[0, 1]`, and stay in it. Most transforms work in 2D and 3D; the few that
only make sense for 2D images say so.
"""

import math

import torch
import torch.nn.functional as F

from torchtransform import functional
from torchtransform.distributions import Parameter, symmetric_log, to_distribution
from torchtransform.state import Context
from torchtransform.transform import PixelTransform

# The standard JPEG quantization table of luminance (ITU-T T.81, Annex K).
JPEG_LUMINANCE_TABLE: tuple[tuple[int, ...], ...] = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

# The standard JPEG quantization table of chrominance (ITU-T T.81, Annex K).
JPEG_CHROMINANCE_TABLE: tuple[tuple[int, ...], ...] = (
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
)

# Fills of `Erasing` that are not a value.
ERASING_FILLS: tuple[str, ...] = ("noise", "mean")


def _require_2d(transform: PixelTransform, image: torch.Tensor) -> None:
    """Raises unless images are 2D."""

    if image.ndim != 4:
        raise ValueError(f"{type(transform).__name__} is only available in 2D.")


def _centered_positions(image: torch.Tensor) -> list[torch.Tensor]:
    """Returns the pixel centers along every spatial axis, relative to the center, broadcastable."""

    spatial = image.shape[2:]
    ndim = len(spatial)

    positions = []
    for axis, size in enumerate(spatial):
        view = [1] * (ndim + 2)
        view[2 + axis] = size
        values = torch.arange(size, device=image.device, dtype=image.dtype) + 0.5 - 0.5 * size
        positions.append(values.view(view))

    return positions


def _renormalize(images: torch.Tensor) -> torch.Tensor:
    """Stretches every element in-place to span `[0, 1]` over all its channels."""

    dims = tuple(range(1, images.ndim))
    low = images.amin(dim=dims, keepdim=True)
    high = images.amax(dim=dims, keepdim=True)

    return images.sub_(low).div_(high - low + 1e-6)


class Gamma(PixelTransform):
    def __init__(self, gamma: Parameter = (0.7, 1.4), p: float = 1.0) -> None:
        """Bends the tone curve, `x` becoming `x ** gamma`.

        Args:
            gamma: The exponent, drawn uniformly in the logarithm. Above one darkens the midtones,
                below one lifts them. A single number `a` is the range `(1 / a, a)`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.gamma = to_distribution(symmetric_log(gamma), log=True)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(gamma=context.sample(self.gamma))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        gamma = functional.per_element(parameters["gamma"], image)

        return image.clamp_(min=0).pow_(gamma)


class Solarize(PixelTransform):
    def __init__(self, threshold: Parameter = 0.5, p: float = 1.0) -> None:
        """Inverts the values at or above a threshold.

        Args:
            threshold: The threshold in `[0, 1]`.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.threshold = to_distribution(threshold)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(threshold=context.sample(self.threshold))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        threshold = functional.per_element(parameters["threshold"], image)

        return torch.where(image >= threshold, 1 - image, image)


class Posterize(PixelTransform):
    def __init__(self, bits: Parameter = (4, 8), p: float = 1.0) -> None:
        """Keeps only the highest bits of every value, as of an 8-bit image.

        Args:
            bits: Number of bits to keep, from one to eight, both bounds of a range included.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.bits = to_distribution(bits, integer=True)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(bits=context.sample(self.bits).clamp(1, 8))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        step = functional.per_element(2 ** (8 - parameters["bits"]), image)
        levels = torch.round(image.clamp(0, 1) * 255)

        return torch.floor(levels / step) * step / 255


class Equalize(PixelTransform):
    def __init__(self, p: float = 1.0) -> None:
        """Equalizes the histogram of every channel, on 256 levels.

        Args:
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        batch_size, channels = image.shape[:2]

        levels = torch.round(image.clamp(0, 1) * 255).to(dtype=torch.int64)
        flat = levels.reshape(batch_size * channels, -1)
        count = flat.shape[1]

        histograms = torch.zeros(
            batch_size * channels, 256, device=image.device, dtype=torch.float64
        )
        histograms.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float64))

        # The cumulative histogram, stretched so the lowest level present maps to zero.
        cumulative = histograms.cumsum(dim=1)
        count_tensor = torch.full_like(cumulative, float(count))
        lowest = torch.where(histograms > 0, cumulative, count_tensor).amin(dim=1, keepdim=True)
        spread = count - lowest

        tables = (cumulative - lowest) / spread.clamp(min=1)
        equalized = torch.gather(tables, 1, flat).to(dtype=image.dtype).view(image.shape)

        # A channel of a single level is left alone.
        flat_channel = (spread == 0).view(batch_size, channels, *(1,) * (image.ndim - 2))

        return torch.where(flat_channel, image, equalized)


class Sharpen(PixelTransform):
    def __init__(
        self, amount: Parameter = (0.5, 1.5), sigma: Parameter = (0.5, 1.5), p: float = 1.0
    ) -> None:
        """Sharpens with an unsharp mask, adding back what a blur takes away.

        Args:
            amount: How much of the detail is added: zero leaves the image alone.
            sigma: Standard deviation of the blur in pixels, the scale of the detail.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.amount = to_distribution(amount)
        self.sigma = to_distribution(sigma)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(amount=context.sample(self.amount), sigma=context.sample(self.sigma))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        blurred = functional.gaussian_blur(image, parameters["sigma"])
        amount = functional.per_element(parameters["amount"], image)

        return (image + amount * (image - blurred)).clamp_(0, 1)


class GaussianBlur(PixelTransform):
    def __init__(self, sigma: Parameter = (0.1, 2.0), p: float = 1.0) -> None:
        """Blurs with a Gaussian.

        Args:
            sigma: Standard deviation in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.sigma = to_distribution(sigma)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(sigma=context.sample(self.sigma))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        return functional.gaussian_blur(image, parameters["sigma"])


class MotionBlur(PixelTransform):
    def __init__(
        self, length: Parameter = (3.0, 9.0), angle: Parameter = (0.0, 180.0), p: float = 1.0
    ) -> None:
        """Blurs along a line, as a camera or a scanned page moving does. 2D only.

        Args:
            length: Length of the line in pixels.
            angle: Direction of the line in degrees, from the x axis toward the y axis.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.length = to_distribution(length)
        self.angle = to_distribution(angle)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(
            length=context.sample(self.length).clamp(min=1), angle=context.sample(self.angle)
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _require_2d(self, image)

        length = parameters["length"].to(dtype=torch.float64)
        radians = torch.deg2rad(parameters["angle"].to(dtype=torch.float64))

        radius = int(math.ceil(0.5 * float(length.max())))
        offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float64)
        y, x = torch.meshgrid(offsets, offsets, indexing="ij")

        cos, sin = torch.cos(radians)[:, None, None], torch.sin(radians)[:, None, None]

        # The distance of every kernel pixel to the line segment, which is antialiased by
        # weighting pixels by how close they are to it.
        along = x * cos + y * sin
        across = -x * sin + y * cos
        beyond = (along.abs() - 0.5 * length[:, None, None] + 0.5).clamp(min=0)
        kernels = (1 - torch.sqrt(across**2 + beyond**2)).clamp(min=0)
        kernels = kernels / kernels.sum(dim=(1, 2), keepdim=True)

        return functional.convolve(image, kernels)


class Defocus(PixelTransform):
    def __init__(self, radius: Parameter = (1.0, 4.0), p: float = 1.0) -> None:
        """Blurs with a disk, as a lens out of focus does. 2D only.

        Args:
            radius: Radius of the disk in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.radius = to_distribution(radius)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(radius=context.sample(self.radius).clamp(min=0))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _require_2d(self, image)

        radius = parameters["radius"].to(dtype=torch.float64)[:, None, None]

        size = int(math.ceil(float(radius.max()) + 0.5))
        offsets = torch.arange(-size, size + 1, device=image.device, dtype=torch.float64)
        y, x = torch.meshgrid(offsets, offsets, indexing="ij")

        # The edge of the disk is antialiased over one pixel.
        kernels = (radius + 0.5 - torch.sqrt(x**2 + y**2)).clamp(0, 1)
        kernels = kernels / kernels.sum(dim=(1, 2), keepdim=True)

        return functional.convolve(image, kernels)


class GaussianNoise(PixelTransform):
    def __init__(
        self, std: Parameter = (0.0, 0.05), per_channel: bool = True, p: float = 1.0
    ) -> None:
        """Adds Gaussian noise.

        Args:
            std: Standard deviation of the noise.
            per_channel: Whether every channel gets its own noise, which gives colored noise,
                rather than the same noise in all channels.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.std = to_distribution(std)
        self.per_channel = per_channel

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(std=context.sample(self.std))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        shape = image.shape if self.per_channel else (image.shape[0], 1, *image.shape[2:])
        generator = context.device_generator(image.device)
        noise = torch.randn(shape, generator=generator, device=image.device, dtype=image.dtype)

        std = functional.per_element(parameters["std"], image)

        return image.add_(noise * std).clamp_(0, 1)


class PoissonNoise(PixelTransform):
    def __init__(self, photons: Parameter = (30.0, 1000.0), p: float = 1.0) -> None:
        """Adds shot noise: a pixel counts photons, as many as its value times the photons at white.

        Args:
            photons: Expected photon count at white, drawn uniformly in the logarithm. Fewer is
                noisier.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.photons = to_distribution(photons, log=True)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(photons=context.sample(self.photons))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        photons = functional.per_element(parameters["photons"], image)
        generator = context.device_generator(image.device)

        counts = torch.poisson(image.clamp(0, 1) * photons, generator=generator)

        return (counts / photons).clamp_(0, 1)


class SpeckleNoise(PixelTransform):
    def __init__(self, std: Parameter = (0.0, 0.1), p: float = 1.0) -> None:
        """Multiplies by noise around one, `x` becoming `x * (1 + std * n)`.

        Args:
            std: Standard deviation of the noise.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.std = to_distribution(std)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(std=context.sample(self.std))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        generator = context.device_generator(image.device)
        noise = torch.randn(
            image.shape, generator=generator, device=image.device, dtype=image.dtype
        )

        std = functional.per_element(parameters["std"], image)

        return image.mul_(1 + std * noise).clamp_(0, 1)


class SaltAndPepperNoise(PixelTransform):
    def __init__(
        self, amount: Parameter = (0.0, 0.05), salt: Parameter = 0.5, p: float = 1.0
    ) -> None:
        """Turns random pixels white (salt) or black (pepper), all their channels together.

        Args:
            amount: Fraction of the pixels that is changed.
            salt: Fraction of the changed pixels that turns white.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.amount = to_distribution(amount)
        self.salt = to_distribution(salt)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(amount=context.sample(self.amount), salt=context.sample(self.salt))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        generator = context.device_generator(image.device)
        shape = (image.shape[0], 1, *image.shape[2:])
        draws = torch.rand(shape, generator=generator, device=image.device, dtype=image.dtype)

        amount = functional.per_element(parameters["amount"], image)
        salted = amount * functional.per_element(parameters["salt"], image)

        image = torch.where(draws < salted, torch.ones_like(image), image)

        return torch.where((draws >= salted) & (draws < amount), torch.zeros_like(image), image)


class JpegCompression(PixelTransform):
    def __init__(
        self, quality: Parameter = (30, 95), subsampling: bool = True, p: float = 1.0
    ) -> None:
        """Adds the artifacts of JPEG compression: blocking, ringing and color bleeding. 2D only.

        The compression is done the way a JPEG encoder does it, in torch: RGB to YCbCr, chroma
        subsampled by two, a discrete cosine transform of every 8 by 8 block, and quantization by
        the standard tables scaled for the quality. Images with another number of channels than
        three have every channel compressed as luminance.

        Args:
            quality: Quality from one to 100, both bounds of a range included. At 100 the result
                is nearly the image itself.
            subsampling: Whether RGB images have their chroma subsampled by two (4:2:0).
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.quality = to_distribution(quality, integer=True)
        self.subsampling = subsampling

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(quality=context.sample(self.quality).clamp(1, 100))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _require_2d(self, image)

        batch_size, channels, height, width = image.shape
        device = image.device

        # Worked on in float32 at least, on the 0 to 255 scale of an 8-bit image.
        work_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        values = image.to(dtype=work_dtype).clamp(0, 1) * 255

        rgb = channels == 3
        subsample = rgb and self.subsampling
        multiple = 16 if subsample else 8

        pad_height, pad_width = -height % multiple, -width % multiple
        values = F.pad(values, (0, pad_width, 0, pad_height), mode="replicate")

        quality = parameters["quality"].to(dtype=work_dtype)
        luminance = _get_jpeg_tables(JPEG_LUMINANCE_TABLE, quality)
        chrominance = _get_jpeg_tables(JPEG_CHROMINANCE_TABLE, quality)
        basis = _get_dct_basis(device, work_dtype)

        if not rgb:
            planes = [
                _compress_plane(values[:, c : c + 1], luminance, basis) for c in range(channels)
            ]
            output = torch.cat(planes, dim=1)
        else:
            red, green, blue = values.unbind(dim=1)

            y = 0.299 * red + 0.587 * green + 0.114 * blue
            cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128
            cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128

            y = _compress_plane(y[:, None], luminance, basis)

            chroma = torch.stack((cb, cr), dim=1)
            if subsample:
                chroma = F.avg_pool2d(chroma, 2)

            cb = _compress_plane(chroma[:, :1], chrominance, basis)
            cr = _compress_plane(chroma[:, 1:], chrominance, basis)

            chroma = torch.cat((cb, cr), dim=1)
            if subsample:
                chroma = F.interpolate(
                    chroma, size=y.shape[2:], mode="bilinear", align_corners=False
                )

            y, cb, cr = y[:, 0], chroma[:, 0] - 128, chroma[:, 1] - 128

            output = torch.stack(
                (y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), dim=1
            )

        # Decoded as 8-bit values.
        output = torch.round(output[:, :, :height, :width].clamp(0, 255)) / 255

        return output.to(dtype=image.dtype)


def _get_jpeg_tables(table: tuple[tuple[int, ...], ...], quality: torch.Tensor) -> torch.Tensor:
    """Returns a quantization table scaled per quality as libjpeg does, of shape `(b, 8, 8)`."""

    base = torch.tensor(table, device=quality.device, dtype=quality.dtype)
    scale = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality)[:, None, None]

    return torch.floor((base * scale + 50) / 100).clamp(1, 255)


def _get_dct_basis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns the orthonormal 8 by 8 DCT-II matrix, rows the frequencies."""

    positions = torch.arange(8, device=device, dtype=dtype)
    frequencies = positions[:, None]

    basis = torch.cos(math.pi * (2 * positions[None] + 1) * frequencies / 16)
    basis[0] *= math.sqrt(1 / 8)
    basis[1:] *= math.sqrt(2 / 8)

    return basis


def _compress_plane(plane: torch.Tensor, tables: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Quantizes the DCT of every 8 by 8 block of a plane.

    Args:
        plane: Plane of shape `(b, 1, H, W)`, with `H` and `W` multiples of eight, in `[0, 255]`.
        tables: Quantization tables of shape `(b, 8, 8)`.
        basis: The DCT matrix.

    Returns:
        The decoded plane, of the same shape.
    """

    batch_size, _, height, width = plane.shape

    blocks = (
        (plane[:, 0] - 128).view(batch_size, height // 8, 8, width // 8, 8).permute(0, 1, 3, 2, 4)
    )

    coefficients = basis @ blocks @ basis.T

    steps = tables[:, None, None].to(dtype=plane.dtype)
    coefficients = torch.round(coefficients / steps) * steps

    blocks = basis.T @ coefficients @ basis

    return (blocks.permute(0, 1, 3, 2, 4).reshape(batch_size, 1, height, width)) + 128


def _resample_coarse(image: torch.Tensor, factors: torch.Tensor, mode: str) -> torch.Tensor:
    """Samples images down to a coarse grid and back up, at a factor per element.

    Args:
        image: Images of shape `(b, C, ...)`.
        factors: Factors of shape `(b,)` in `(0, 1]`.
        mode: How to go back up: `"bilinear"` or `"nearest"`.

    Returns:
        The images, of the same shape.
    """

    spatial = image.shape[2:]
    device, dtype = image.device, image.dtype

    factors = factors.to(dtype=torch.float64).clamp(1e-3, 1)

    # A cell of `1 / f` pixels averages them: a box of that width has variance `w ** 2 / 12`, of
    # which a pixel itself already covers `1 / 12`.
    sigmas = torch.sqrt(((1 / factors) ** 2 - 1).clamp(min=0) / 12)
    blurred = functional.gaussian_blur(image, sigmas.to(dtype=dtype))

    counts = [torch.round(size * factors).clamp(min=1) for size in spatial]
    largest = [int(c.max()) for c in counts]

    # Down: every coarse sample at the center of its cell.
    down_axes = []
    for axis, count in enumerate(counts):
        positions = torch.arange(largest[axis], device=device, dtype=torch.float64)
        normalized = ((positions[None] + 0.5) / count.to(device)[:, None] * 2 - 1).clamp(max=1)
        down_axes.append(normalized)

    coarse = F.grid_sample(
        blurred,
        _separable_grid(down_axes, dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    # Up: every pixel from where it lies on the coarse grid of its element.
    up_axes = []
    for axis, (size, count) in enumerate(zip(spatial, counts)):
        positions = torch.arange(size, device=device, dtype=torch.float64)[None] + 0.5
        count = count.to(device)[:, None]

        if mode == "nearest":
            coordinates = torch.floor(positions * count / size).clamp(max=count - 1)
        else:
            coordinates = (positions * count / size - 0.5).clamp(min=0).minimum(count - 1)

        span = max(largest[axis] - 1, 1)
        up_axes.append(coordinates / span * 2 - 1)

    return F.grid_sample(
        coarse,
        _separable_grid(up_axes, dtype),
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )


def _separable_grid(axes: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """Returns a sampling grid of shape `(b, *S, n)` from normalized positions per axis.

    Args:
        axes: Positions of shape `(b, S_i)` for every axis, in tensor order.
        dtype: Floating point type of the grid.
    """

    batch_size = axes[0].shape[0]
    shape = tuple(a.shape[1] for a in axes)
    ndim = len(axes)

    grids = []
    for axis, positions in enumerate(axes):
        view = [batch_size] + [1] * ndim
        view[1 + axis] = shape[axis]
        grids.append(positions.view(view).expand(batch_size, *shape))

    return torch.stack(grids[::-1], dim=-1).to(dtype=dtype)


class Downscale(PixelTransform):
    def __init__(
        self, factor: Parameter = (0.25, 0.75), mode: str = "bilinear", p: float = 1.0
    ) -> None:
        """Lowers the resolution: shrinks, then enlarges back to the shape it had.

        Args:
            factor: Factor it is shrunk by, in `(0, 1]`.
            mode: How it is enlarged: `"bilinear"` (soft) or `"nearest"` (blocky).
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        if mode not in ("bilinear", "nearest"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'bilinear' or 'nearest'.")

        self.factor = to_distribution(factor)
        self.mode = mode

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(factor=context.sample(self.factor))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        return _resample_coarse(image, parameters["factor"], self.mode)


class Pixelate(PixelTransform):
    def __init__(self, cell: Parameter = (4.0, 12.0), p: float = 1.0) -> None:
        """Replaces the image by square cells of their average color.

        Args:
            cell: Side of a cell in pixels.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.cell = to_distribution(cell)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(cell=context.sample(self.cell).clamp(min=1))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        return _resample_coarse(image, 1 / parameters["cell"], "nearest")


class Erasing(PixelTransform):
    def __init__(
        self,
        count: Parameter = (1, 3),
        size: Parameter = (0.05, 0.25),
        fill: float | tuple[float, ...] | str = 0.0,
        p: float = 1.0,
    ) -> None:
        """Erases random boxes, as cutout, random erasing or coarse dropout do.

        Args:
            count: Number of boxes, both bounds of a range included.
            size: Side of a box along every axis, drawn per axis, as a fraction of the canvas.
            fill: What the boxes are filled with: a value (one, or one per channel), `"noise"` for
                uniform noise, or `"mean"` for the mean color of the element.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        if isinstance(fill, str) and fill not in ERASING_FILLS:
            raise ValueError(f"Unknown fill {fill!r}, expected a value or one of {ERASING_FILLS}.")

        self.count = to_distribution(count, integer=True)
        self.size = to_distribution(size)
        self.fill = fill

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        most = max(int(self.count.high), 0)

        return dict(
            count=context.sample(self.count).clamp(min=0),
            centers=context.rand(context.batch_size, most, context.ndim),
            sizes=context.sample(self.size, most, context.ndim),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        batch_size, channels = image.shape[:2]
        spatial = image.shape[2:]
        ndim = len(spatial)

        centers, sizes = parameters["centers"], parameters["sizes"]
        most = centers.shape[1]
        if most == 0:
            return image

        # Per box and axis, which pixels lie within it, combined over the axes.
        inside = torch.arange(most, device=image.device)[None] < parameters["count"][:, None]
        inside = inside.view(batch_size, most, *(1,) * ndim)
        for axis, size in enumerate(spatial):
            positions = (torch.arange(size, device=image.device, dtype=image.dtype) + 0.5) / size
            within = (positions[None, None] - centers[..., axis, None]).abs() <= 0.5 * sizes[
                ..., axis, None
            ]

            view = [batch_size, most] + [1] * ndim
            view[2 + axis] = size
            inside = inside & within.view(view)

        erased = inside.any(dim=1, keepdim=True)

        if self.fill == "noise":
            generator = context.device_generator(image.device)
            fill = torch.rand(
                image.shape, generator=generator, device=image.device, dtype=image.dtype
            )
        elif self.fill == "mean":
            fill = image.mean(dim=functional.spatial_dims(image), keepdim=True)
        else:
            values = torch.as_tensor(self.fill, device=image.device, dtype=image.dtype).flatten()
            fill = values.expand(channels).view(1, channels, *(1,) * ndim)

        return torch.where(erased, fill, image)


class Vignette(PixelTransform):
    def __init__(
        self, strength: Parameter = (0.2, 0.6), radius: Parameter = (0.4, 0.8), p: float = 1.0
    ) -> None:
        """Darkens toward the corners, as a lens does.

        Args:
            strength: How dark the corners go: zero leaves the image alone, one takes them black.
            radius: Distance from the center where the darkening starts, as a fraction of half the
                diagonal.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.strength = to_distribution(strength)
        self.radius = to_distribution(radius)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(strength=context.sample(self.strength), radius=context.sample(self.radius))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        positions = _centered_positions(image)
        squared = sum(x * x for x in positions)
        half_diagonal = 0.5 * math.sqrt(sum(s * s for s in image.shape[2:]))
        distances = torch.sqrt(squared) / half_diagonal

        strength = functional.per_element(parameters["strength"], image)
        radius = functional.per_element(parameters["radius"], image).clamp(max=0.999)

        falloff = ((distances - radius) / (1 - radius)).clamp(0, 1) ** 2

        return functional.darken(image, 1 - strength * falloff)


class BackgroundIrregularities(PixelTransform):
    def __init__(
        self,
        intensity: Parameter = (0.01, 0.05),
        scale: int | tuple[int, ...] | None = None,
        p: float = 1.0,
    ) -> None:
        """Adds smooth irregularities, such as smudges and uneven lighting, and restretches.

        The image is stretched back to span `[0, 1]` afterward.

        Args:
            intensity: Strength of the irregularities.
            scale: Cells of the coarse grid the irregularities are drawn on, along every axis in
                tensor order (such as `(H, W)`) or one number for all. Fewer cells are larger
                smudges. By default `(8, 6)` in 2D and `(6, 8, 6)` in 3D.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.intensity = to_distribution(intensity)
        self.scale = scale

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(intensity=context.sample(self.intensity))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        batch_size = image.shape[0]
        spatial = image.shape[2:]
        ndim = len(spatial)

        if self.scale is None:
            grid = (8, 6) if ndim == 2 else (6, 8, 6)
        elif isinstance(self.scale, int):
            grid = (self.scale,) * ndim
        else:
            grid = tuple(self.scale)

        if len(grid) != ndim:
            raise ValueError(f"The scale {self.scale} does not fit {ndim}D images.")

        generator = context.device_generator(image.device)
        noise = torch.randn(
            (batch_size, 1, *grid), generator=generator, device=image.device, dtype=image.dtype
        )
        mode = "bicubic" if ndim == 2 else "trilinear"
        smudge = F.interpolate(noise, size=spatial, mode=mode, align_corners=False)

        # A sign and a power for more natural smudge shapes.
        smudge = smudge.sign() * smudge.abs().pow(0.8)
        smudge = smudge * functional.per_element(parameters["intensity"], image)

        return _renormalize(image.add_(smudge))


class BackgroundGradient(PixelTransform):
    def __init__(
        self,
        intensity: Parameter = (0.1, 0.4),
        length: Parameter = (0.3, 1.0),
        p: float = 1.0,
    ) -> None:
        """Adds a linear gradient in a random direction, as of uneven lighting, and restretches.

        The image is stretched back to span `[0, 1]` afterward.

        Args:
            intensity: Strength of the gradient.
            length: Relative length of the gradient: zero is none, one spans the whole image.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.intensity = to_distribution(intensity)
        self.length = to_distribution(length)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(
            intensity=context.sample(self.intensity),
            length=context.sample(self.length),
            direction=torch.nn.functional.normalize(
                context.randn(context.batch_size, context.ndim), dim=1
            ),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        batch_size = image.shape[0]
        spatial = image.shape[2:]
        ndim = len(spatial)
        direction = parameters["direction"]

        # Project every pixel on the direction, in coordinates from zero to one per axis.
        projection = torch.zeros((batch_size, 1, *spatial), device=image.device, dtype=image.dtype)
        for axis, size in enumerate(spatial):
            view = [1] * (ndim + 2)
            view[2 + axis] = size
            positions = torch.linspace(0, 1, size, device=image.device, dtype=image.dtype).view(
                view
            )

            component = direction[:, ndim - 1 - axis].view(-1, *(1,) * (ndim + 1))
            projection = projection + positions * component

        # Normalized to [-1, 1] per element.
        dims = tuple(range(1, image.ndim))
        low = projection.amin(dim=dims, keepdim=True)
        high = projection.amax(dim=dims, keepdim=True)
        projection = 2 * (projection - low) / (high - low + 1e-6) - 1

        scale = parameters["length"] * parameters["intensity"]
        gradient = projection * functional.per_element(scale, image)

        return _renormalize(image.add_(gradient))


class Clamp(PixelTransform):
    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        """Clamps the values to a range.

        Args:
            low: Lowest value.
            high: Highest value.
        """

        super().__init__()

        self.low = float(low)
        self.high = float(high)

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        return image.clamp_(self.low, self.high)


class ChromaticAberration(PixelTransform):
    def __init__(self, shift: Parameter = (0.0, 2.0), p: float = 1.0) -> None:
        """Fringes edges with color, as a lens that bends red and blue differently does.

        The red channel is enlarged and the blue channel shrunk around the center, so that at the
        corners they move by `shift` pixels in opposite directions and the center stays sharp.
        2D RGB only.

        Args:
            shift: How far the red and blue channels move at the corners, in pixels. A negative
                shift fringes the other way.
            p: Probability that it is applied, per element.
        """

        super().__init__(p)

        self.shift = to_distribution(shift)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        radius = 0.5 * torch.linalg.norm(context.size)

        return dict(scale=context.sample(self.shift) / radius)

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _require_2d(self, image)

        if image.shape[1] != 3:
            raise ValueError(
                f"ChromaticAberration needs RGB images, not {image.shape[1]} channels."
            )

        batch_size = image.shape[0]
        scale = parameters["scale"]

        output = image.clone()
        for channel, sign in ((0, 1.0), (2, -1.0)):
            # A pixel of the enlarged channel shows what lies closer to the center.
            factors = 1 / (1 + sign * scale)
            theta = torch.zeros(batch_size, 2, 3, device=image.device, dtype=image.dtype)
            theta[:, 0, 0] = factors
            theta[:, 1, 1] = factors

            grid = F.affine_grid(theta, [batch_size, 1, *image.shape[2:]], align_corners=False)
            output[:, channel : channel + 1] = F.grid_sample(
                image[:, channel : channel + 1], grid, padding_mode="border", align_corners=False
            )

        return output
