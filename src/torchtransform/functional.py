"""Functions on batches of images that the transforms are built from.

Images are tensors of shape `(B, C, H, W)` or `(B, C, D, H, W)`. Where a function takes a
parameter per element, it is a tensor of shape `(B,)`, so every element can be treated differently
in one batched operation.
"""

import math

import torch
import torch.nn.functional as F

# Weights of the red, green and blue channels in the luma of an image (ITU-R BT.601).
LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)


def spatial_dims(images: torch.Tensor) -> tuple[int, ...]:
    """Returns the spatial dimensions of images, all but the batch and channel dimensions."""

    return tuple(range(2, images.ndim))


def per_element(values: torch.Tensor | float, images: torch.Tensor) -> torch.Tensor:
    """Reshapes a value per element, of shape `(B,)`, to broadcast against images."""

    values = torch.as_tensor(values, device=images.device, dtype=images.dtype)
    if values.ndim == 0:
        return values

    return values.view(-1, *(1,) * (images.ndim - 1))


def rgb_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Returns the luma of RGB images, of shape `(B, 1, ...)`."""

    if images.shape[1] != 3:
        raise ValueError(f"Expected RGB images with three channels, not {images.shape[1]}.")

    red, green, blue = images.unbind(dim=1)
    weights = LUMA_WEIGHTS

    return (weights[0] * red + weights[1] * green + weights[2] * blue).unsqueeze(1)


def to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Returns the luma of RGB images, or the channel mean of others, of shape `(B, 1, ...)`."""

    if images.shape[1] == 3:
        return rgb_to_grayscale(images)

    return images.mean(dim=1, keepdim=True)


def rgb_to_hsv(images: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Converts RGB images to HSV, all channels in `[0, 1]`."""

    maximum, argmax = images.max(dim=1)
    minimum = images.amin(dim=1)
    delta = maximum - minimum

    value = maximum
    saturation = delta / (maximum + epsilon)

    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    red, green, blue = (maximum.unsqueeze(1) - images).unbind(dim=1)

    hues = torch.stack((blue - green, (red - blue) + 2 * delta, (green - red) + 4 * delta), dim=1)
    hue = torch.gather(hues / delta.unsqueeze(1), 1, argmax.unsqueeze(1)).squeeze(1)
    hue = (hue / 6) % 1

    return torch.stack((hue, saturation, value), dim=1)


def hsv_to_rgb(images: torch.Tensor) -> torch.Tensor:
    """Converts HSV images, all channels in `[0, 1]`, to RGB."""

    hue, saturation, value = images.unbind(dim=1)

    sector = torch.floor(hue * 6) % 6
    fraction = (hue * 6 % 6) - sector

    chroma = saturation * value
    p = value - chroma
    q = value - chroma * fraction
    t = p + chroma * fraction

    sector = sector.long()
    indices = torch.stack((sector, sector + 6, sector + 12), dim=1)
    choices = torch.stack(
        (value, q, p, p, t, value, t, value, value, q, p, p, p, p, t, value, value, q), dim=1
    )

    return torch.gather(choices, 1, indices)


def gaussian_kernel_size(sigmas: torch.Tensor, truncate: float = 2.0) -> int:
    """Returns the odd kernel size that covers the largest sigma up to `truncate` of them."""

    return 2 * int(math.ceil(truncate * float(sigmas.max()))) + 1 if len(sigmas) else 1


def gaussian_blur(
    images: torch.Tensor, sigmas: torch.Tensor | float, truncate: float = 2.0
) -> torch.Tensor:
    """Blurs images with a Gaussian, of its own standard deviation per element.

    Every element is convolved with a kernel as wide as the largest sigma needs, so this stays a
    single batched convolution. Small 2D kernels are applied as one square kernel, which is faster
    than two one-dimensional passes on large images because it writes no intermediate image; larger
    kernels, and all 3D kernels, are applied one axis at a time.

    Args:
        images: Tensor of shape `(B, C, H, W)` or `(B, C, D, H, W)`.
        sigmas: Standard deviations in pixels, of shape `(B,)`, or one for all.
        truncate: Where the kernel is cut off, in standard deviations.

    Returns:
        The blurred images, of the same shape. Borders are reflected.
    """

    batch_size, channels = images.shape[:2]
    spatial = images.shape[2:]
    ndim = len(spatial)

    sigmas = torch.as_tensor(sigmas, device=images.device, dtype=images.dtype).flatten()
    sigmas = sigmas.expand(batch_size).clamp(min=1e-3)

    size = gaussian_kernel_size(sigmas, truncate)
    radius = (size - 1) // 2
    if radius == 0:
        return images

    positions = torch.arange(size, device=images.device, dtype=images.dtype) - radius
    kernels = torch.exp(-(positions[None] ** 2) / (2 * sigmas[:, None] ** 2))
    kernels = kernels / kernels.sum(dim=1, keepdim=True)

    # Reflect padding needs the padding to be smaller than the image.
    pad_mode = "reflect" if all(radius < s for s in spatial) else "replicate"
    flat = images.reshape(1, batch_size * channels, *spatial)

    if ndim == 2 and size <= 9:
        weight = (kernels[:, :, None] * kernels[:, None, :]).repeat_interleave(channels, dim=0)
        padded = F.pad(flat, (radius,) * 4, mode=pad_mode)

        return F.conv2d(padded, weight[:, None], groups=batch_size * channels).view(images.shape)

    convolution = F.conv2d if ndim == 2 else F.conv3d
    weight = kernels.repeat_interleave(channels, dim=0)

    output = flat
    for axis in range(ndim):
        shape = [batch_size * channels, 1] + [1] * ndim
        shape[2 + axis] = size

        padding = [0] * (2 * ndim)
        padding[2 * (ndim - 1 - axis)] = radius
        padding[2 * (ndim - 1 - axis) + 1] = radius

        output = F.pad(output, padding, mode=pad_mode)
        output = convolution(output, weight.view(shape), groups=batch_size * channels)

    return output.view(images.shape)


def convolve(
    images: torch.Tensor, kernels: torch.Tensor, pad_mode: str = "reflect"
) -> torch.Tensor:
    """Convolves every element of a batch of 2D images with its own kernel.

    Args:
        images: Tensor of shape `(B, C, H, W)`.
        kernels: Kernels of shape `(B, K, K)` with `K` odd, applied to every channel.
        pad_mode: How borders are padded, as for `torch.nn.functional.pad`.

    Returns:
        The convolved images, of the same shape.
    """

    batch_size, channels, height, width = images.shape
    size = kernels.shape[-1]
    radius = size // 2

    if radius >= min(height, width) and pad_mode == "reflect":
        pad_mode = "replicate"

    weight = kernels.to(dtype=images.dtype, device=images.device).repeat_interleave(channels, dim=0)
    flat = F.pad(
        images.reshape(1, batch_size * channels, height, width), (radius,) * 4, mode=pad_mode
    )

    return F.conv2d(flat, weight[:, None], groups=batch_size * channels).view(images.shape)


def max_filter(images: torch.Tensor, size: int) -> torch.Tensor:
    """Returns the largest value within a square or cube of an odd side around every pixel.

    The square is taken one axis at a time, which gives the same largest value at a cost that grows
    with the side rather than with its square.
    """

    if size % 2 == 0:
        raise ValueError("The size must be odd.")

    ndim = images.ndim - 2
    pool = F.max_pool2d if ndim == 2 else F.max_pool3d

    for axis in range(ndim):
        kernel = [1] * ndim
        kernel[axis] = size
        padding = [0] * ndim
        padding[axis] = size // 2

        images = pool(images, tuple(kernel), stride=1, padding=tuple(padding))

    return images


def erode(images: torch.Tensor, size: int) -> torch.Tensor:
    """Grows the dark parts of images by `size // 2` pixels, an erosion of intensity."""

    return -max_filter(-images, size)


def dilate(images: torch.Tensor, size: int) -> torch.Tensor:
    """Grows the light parts of images by `size // 2` pixels, a dilation of intensity."""

    return max_filter(images, size)


def value_noise(
    shape: tuple[int, ...],
    cell_size: torch.Tensor | float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    normal: bool = False,
) -> torch.Tensor:
    """Draws noise on a coarse grid of cells and interpolates it smoothly to the full size.

    Clumped noise like this is what makes blotches, stains and patchy ink look natural rather than
    like per-pixel dropout.

    Args:
        shape: Shape `(B, C, ...)` of the noise.
        cell_size: Side of a cell in pixels, roughly the size of a clump. One for all elements, or
            one per element, of shape `(B,)`.
        generator: Generator on the device to draw with.
        device: Device to draw on.
        dtype: Floating point type to draw.
        normal: Whether to draw standard normal values rather than uniform values in `[0, 1)`.

    Returns:
        The noise, of the given shape. Uniform noise stays in `[0, 1)`, since the interpolation is
        linear.
    """

    batch_size, channels = shape[:2]
    spatial = tuple(shape[2:])
    ndim = len(spatial)

    cell_sizes = (
        torch.as_tensor(cell_size, dtype=torch.float64).flatten().expand(batch_size).clamp(min=1)
    )
    smallest = float(cell_sizes.min())

    draw = torch.randn if normal else torch.rand
    mode = "bilinear" if ndim == 2 else "trilinear"

    if bool(torch.all(cell_sizes == cell_sizes[0])):
        grid_shape = tuple(max(1, math.ceil(s / smallest)) for s in spatial)
        noise = draw(
            (batch_size, channels, *grid_shape), generator=generator, device=device, dtype=dtype
        )

        return F.interpolate(noise, size=spatial, mode=mode, align_corners=False)

    grid_shape = tuple(max(2, math.ceil(s / smallest) + 1) for s in spatial)
    noise = draw(
        (batch_size, channels, *grid_shape), generator=generator, device=device, dtype=dtype
    )

    # With a cell size per element, every element samples the grid at its own spacing.
    axes = [
        (torch.arange(s, device=device, dtype=torch.float64) + 0.5)[None]
        / cell_sizes.to(device)[:, None]
        for s in spatial
    ]
    grids = []
    for axis, (positions, count) in enumerate(zip(axes, grid_shape)):
        view = [batch_size] + [1] * ndim
        view[1 + axis] = spatial[axis]
        grids.append((positions / (count - 1) * 2 - 1).view(view).expand(batch_size, *spatial))

    grid = torch.stack(grids[::-1], dim=-1).to(dtype=dtype)

    return F.grid_sample(noise, grid, mode="bilinear", padding_mode="border", align_corners=True)


def shift(images: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Rolls every element by its own whole number of pixels, wrapping around.

    Args:
        images: Tensor of shape `(B, C, ...)`.
        offsets: Offsets of shape `(B, n)`, in tensor order, such as `(dy, dx)`.

    Returns:
        The rolled images, of the same shape.
    """

    batch_size = images.shape[0]
    spatial = images.shape[2:]
    offsets = offsets.to(device=images.device, dtype=torch.int64)

    output = images
    for axis, size in enumerate(spatial):
        index = torch.remainder(
            torch.arange(size, device=images.device)[None] - offsets[:, axis, None], size
        )

        view = [batch_size] + [1] * (images.ndim - 1)
        view[2 + axis] = size
        index = index.view(view).expand(output.shape)

        output = torch.gather(output, 2 + axis, index)

    return output


def darken(images: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Darkens images in-place by a factor per pixel in `[0, 1]`.

    One leaves a pixel alone, and zero takes it to black.
    """

    return images.mul_(field)


def lighten(images: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Lightens images in-place by a weight per pixel in `[0, 1]`.

    Zero leaves a pixel alone, and one takes it to white.
    """

    return images.mul_(1 - field).add_(field)
