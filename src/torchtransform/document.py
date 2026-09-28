"""Augmentations that make pages look scanned, faxed, photocopied or printed.

They are pixel effects on RGB images of shape `(B, 3, H, W)` with values in `[0, 1]`, and never
move anything, so boxes, points and masks are left alone. Every one of them is tuned to keep a
page legible: labels such as boxes, text or rules claim what can be read on the page, so an effect
may fade or roughen ink, but must not erase it or invent it.

Every batch element draws its own parameters, so a batch of pages gets as many different scanners
as it has pages.
"""

import math
from collections.abc import Iterator
from enum import Enum

import torch
import torch.nn.functional as F

from torchtransform.distributions import Parameter, to_distribution
from torchtransform.functional import (
    darken,
    dilate,
    erode,
    gaussian_blur,
    hsv_to_rgb,
    lighten,
    per_element,
    rgb_to_grayscale,
    shift,
    value_noise,
)
from torchtransform.state import Context
from torchtransform.transform import PixelTransform

# Side of the structuring element the ink augmentations grow and shrink strokes by. Fixed at three,
# and deliberately not a parameter: it moves the edge of a stroke by exactly one pixel, so a three
# pixel stroke always keeps a one pixel core and the counter of an `e` cannot be closed. At five, a
# thin stroke can disappear altogether and the page would no longer say what its labels claim.
INK_MORPHOLOGY_SIZE: int = 3

# The most of the paper a bitonal copy's speckle ever turns black, however stained it was. Past
# about half, the specks run together into solid black, and the text in it is gone.
MAX_SPECKLE_DENSITY: float = 0.55

# How much smaller than the page a bitonal copy works out the level of its paper, and the side of
# the window it does so in at that size: a few text heights.
PAPER_SCALE: int = 4
PAPER_WINDOW: int = 31

# The four diagonal directions a color channel can be pulled in, as `(dy, dx)`.
MISALIGNMENT_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class DitherPattern(str, Enum):
    BAYER = "bayer"  # Dispersed threshold matrix; reads as a fax or a laser printer.
    CLUSTERED = "clustered"  # Round dots that merge into a checkerboard; a classical halftone.
    LINE = "line"  # A line screen, the whole row turning on at once.


# The patterns by index, for drawing a pattern per element.
DITHER_PATTERNS: tuple[DitherPattern, ...] = tuple(DitherPattern)


def bayer_matrix(order: int) -> torch.Tensor:
    """Builds a normalized Bayer threshold matrix.

    The matrix is grown by the usual quadtree recursion, each step replacing every entry with a two
    by two block that spreads the next four threshold levels as far apart as possible.

    Args:
        order: Number of recursion steps, giving a matrix of side `2 ** order`.

    Returns:
        The matrix, of shape `(2 ** order, 2 ** order)`, holding every threshold level in `(0, 1)`
        exactly once.
    """

    if order < 0:
        raise ValueError("Order must be non-negative.")

    matrix = torch.zeros(1, 1, dtype=torch.float64)

    for _ in range(order):
        matrix = torch.cat(
            (
                torch.cat((4 * matrix, 4 * matrix + 2), dim=1),
                torch.cat((4 * matrix + 3, 4 * matrix + 1), dim=1),
            ),
            dim=0,
        )

    return (matrix + 0.5) / matrix.numel()


def rank_tile(priority: torch.Tensor) -> torch.Tensor:
    """Turns a priority map into the thresholds that realize its turn-on order.

    The pixel with the highest priority is given the lowest threshold, so it is the first to take
    ink as the tone darkens.

    Args:
        priority: The priorities, of any shape.

    Returns:
        The thresholds, of the same shape, holding every level in `(0, 1)` exactly once.
    """

    flat = priority.flatten()

    ranks = torch.empty_like(flat)
    ranks[torch.argsort(flat, descending=True)] = torch.arange(
        len(flat), device=flat.device, dtype=flat.dtype
    )

    return ((ranks + 0.5) / len(flat)).view(priority.shape)


def dither_tile(pattern: DitherPattern, size: int) -> torch.Tensor:
    """Builds one period of a dither screen.

    Args:
        pattern: Which screen to build.
        size: Requested number of thresholds along a period. A Bayer matrix only exists at powers
            of two, so there the size is rounded to the nearest one.

    Returns:
        The screen, a square tile of thresholds in `(0, 1)`.
    """

    if size < 2:
        raise ValueError("Size must be at least two.")

    if pattern is DitherPattern.BAYER:
        return bayer_matrix(max(1, round(math.log2(size))))

    coordinates = torch.cos(2 * math.pi * (torch.arange(size, dtype=torch.float64) + 0.5) / size)

    if pattern is DitherPattern.CLUSTERED:
        # The standard spot function. It is periodic in both axes by construction, so the tile
        # repeats without a seam, and its level sets are the round dots of a halftone.
        return rank_tile(-(coordinates[:, None] + coordinates[None, :]))

    # A line screen shares one threshold along the whole row, so a row turns on as a line rather
    # than dissolving into dots.
    return rank_tile(-coordinates)[:, None].expand(size, size).contiguous()


def tile_to(
    tile: torch.Tensor, height: int, width: int, cell: int, offset: tuple[int, int]
) -> torch.Tensor:
    """Repeats a dither tile out to cover an image.

    Args:
        tile: One period of the screen, of shape `(S, S)`.
        height: Height to cover.
        width: Width to cover.
        cell: Pixels per threshold, coarsening the screen without adding levels.
        offset: Phase of the screen in pixels, so it does not always start on the same threshold.

    Returns:
        The thresholds, of shape `(height, width)`.
    """

    if cell > 1:
        tile = tile.repeat_interleave(cell, dim=0).repeat_interleave(cell, dim=1)

    period_height, period_width = tile.shape
    tile = torch.roll(tile, (offset[0] % period_height, offset[1] % period_width), (0, 1))

    repeats_height = -(-height // period_height)
    repeats_width = -(-width // period_width)

    return tile.repeat(repeats_height, repeats_width)[:height, :width]


def open_runs(images: torch.Tensor, length: int, horizontal: bool) -> torch.Tensor:
    """Keeps only what sits inside a long unbroken run, a morphological opening along one axis.

    Args:
        images: Tensor of shape `(B, C, H, W)`, high where something is present.
        length: Length of the run in pixels, which must be odd.
        horizontal: Whether the run is measured along the width or along the height.

    Returns:
        The opened map, of the same shape.
    """

    if length % 2 == 0:
        raise ValueError("Length must be odd.")

    size = (1, length) if horizontal else (length, 1)
    padding = (0, length // 2) if horizontal else (length // 2, 0)

    opened = -F.max_pool2d(-images, size, stride=1, padding=padding)

    return F.max_pool2d(opened, size, stride=1, padding=padding)


def bar_profiles(
    length: int, bar_widths: torch.Tensor, highs: torch.Tensor, lows: torch.Tensor
) -> torch.Tensor:
    """Builds one-dimensional profiles of soft bars, the shape a roller leaves across a page.

    Each bar ramps from its high value down to its low value and back, over twice its width.

    Args:
        length: Length of the profiles in pixels.
        bar_widths: Pixels in half a bar per element, of shape `(B,)`, integer.
        highs: The high value of every bar, of shape `(B, K)`, with enough bars to cover the length.
        lows: The low value of every bar, of shape `(B, K)`.

    Returns:
        The profiles, of shape `(B, length)`.
    """

    positions = torch.arange(length)[None]
    widths = bar_widths[:, None]

    bars = (positions // (2 * widths)).clamp(max=highs.shape[1] - 1)
    within = positions % (2 * widths)

    # The ramp down over the first half of a bar, and back up over the second.
    steps = torch.where(within < widths, within, 2 * widths - 1 - within)
    steps = steps / (widths - 1).clamp(min=1)

    high = torch.gather(highs, 1, bars)
    low = torch.gather(lows, 1, bars)

    return high + (low - high) * steps


def _draw_int(context: Context, value: Parameter, *shape: int) -> torch.Tensor:
    """Draws an integer parameter per element, both bounds of a range included."""

    return context.sample(value, *shape, integer=True).to(dtype=torch.int64)


def _highest(value: Parameter) -> int:
    """Returns the largest whole number an integer parameter can draw."""

    return int(to_distribution(value, integer=True).high)


def _lowest(value: Parameter) -> int:
    """Returns the smallest whole number an integer parameter can draw."""

    return int(to_distribution(value, integer=True).low)


def _groups(keys: torch.Tensor) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields every distinct row of integer keys of shape `(b, k)`, with the indices of its rows."""

    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)

    for i, key in enumerate(unique):
        yield key, torch.nonzero(inverse == i)[:, 0]


def _check(transform: PixelTransform, image: torch.Tensor, rgb: bool) -> None:
    """Raises unless images are 2D, and RGB if needed."""

    if image.ndim != 4:
        raise ValueError(f"{type(transform).__name__} is only available in 2D.")

    if rgb and image.shape[1] != 3:
        raise ValueError(
            f"{type(transform).__name__} needs RGB images, not {image.shape[1]} channels."
        )


def _threshold_maps(
    patterns: torch.Tensor,
    sizes: torch.Tensor,
    cells: torch.Tensor,
    transposed: torch.Tensor,
    offsets: torch.Tensor,
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Builds a dither screen per element, covering an image.

    The same as `tile_to` for every element, with its own pattern, size, cell, direction and phase.

    Args:
        patterns: Index into `DITHER_PATTERNS` per element, of shape `(b,)`.
        sizes: Thresholds along a period, of shape `(b,)`.
        cells: Pixels per threshold, of shape `(b,)`.
        transposed: Whether to transpose the tile, of shape `(b,)`.
        offsets: Phase of the screen in pixels, of shape `(b, 2)`.
        shape: The shape `(H, W)` to cover.
        device: Device of the screens.
        dtype: Floating point type of the screens.

    Returns:
        The thresholds, of shape `(b, 1, H, W)`.
    """

    height, width = shape
    thresholds = torch.empty(len(sizes), 1, height, width, device=device, dtype=dtype)

    keys = torch.stack((patterns, sizes, cells, transposed.to(dtype=torch.int64)), dim=1).cpu()
    offsets = offsets.cpu()

    for key, indices in _groups(keys):
        pattern, size, cell, transpose = (int(x) for x in key)

        tile = dither_tile(DITHER_PATTERNS[pattern], size)
        if transpose:
            tile = tile.t()

        if cell > 1:
            tile = tile.repeat_interleave(cell, dim=0).repeat_interleave(cell, dim=1)

        # A tile rolled by an offset and repeated reads its entry at the position minus the offset.
        period_height, period_width = tile.shape
        rows = torch.remainder(torch.arange(height)[None] - offsets[indices, :1], period_height)
        columns = torch.remainder(torch.arange(width)[None] - offsets[indices, 1:], period_width)

        maps = tile[rows[:, :, None], columns[:, None, :]]
        thresholds[indices.to(device)] = maps[:, None].to(device=device, dtype=dtype)

    return thresholds


class BleedThrough(PixelTransform):
    def __init__(
        self,
        alpha: Parameter = (0.02, 0.18),
        offset: Parameter = (-32, 32),
        sigma: float = 1.2,
        downscale: int = 4,
        p: float = 1.0,
    ) -> None:
        """Shows the ink on the other side of the sheet through the paper.

        The back of a sheet is the page mirrored left to right, never quite registered with the
        front, and seen through the fibers of the paper. The mirror is taken from the page itself,
        which is what a duplex print of the same document would put there.

        Only darkening is applied, so the augmentation can never take ink away. The downscale and
        the blur are not optional either: they are what keep the ghost an unreadable smudge instead
        of legible mirrored text that the labels say nothing about.

        Args:
            alpha: How far the paper is darkened where the back carries the most ink.
            offset: How far the back is shifted against the front, in whole pixels, drawn per axis.
            sigma: Standard deviation of the blur, at the downscaled resolution.
            downscale: Factor the ghost is built at, which is both what makes the blur cheap and
                what dissolves the mirrored strokes.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.alpha = alpha
        self.offset = offset
        self.sigma = sigma
        self.downscale = downscale

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(
            alpha=context.sample(self.alpha),
            offsets=_draw_int(context, self.offset, 2),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        height, width = image.shape[2:]
        factor = self.downscale

        # The sheet seen from the back, shifted because the two sides never line up.
        back = F.avg_pool2d(rgb_to_grayscale(image), factor, ceil_mode=True).flip(-1)
        back = shift(back, torch.div(parameters["offsets"], factor, rounding_mode="floor"))

        # How much ink the back carries, measured against the page's own paper so that a clean
        # sheet stays clean rather than graying out as a whole.
        back = 1 - back
        back = (back - back.mean(dim=(2, 3), keepdim=True)).clamp_min_(0)

        back = gaussian_blur(back, self.sigma)
        back = F.interpolate(back, size=(height, width), mode="bilinear", align_corners=False)

        alpha = per_element(parameters["alpha"], image)

        return darken(image, back.mul_(-alpha).add_(1))


class ChannelMisalignment(PixelTransform):
    def __init__(
        self,
        shift: Parameter = (1, 3),
        mix: Parameter = (0.08, 0.25),
        gain: Parameter = (0.95, 1.05),
        p: float = 1.0,
    ) -> None:
        """Misregisters the color channels, the fringe a scanner's sensor bar leaves on every edge.

        Despite the name this moves no content. Each channel keeps most of its unshifted self, so
        the achromatic edge a box is drawn around stays exactly where it was and only a few pixels
        of color fringe appear beside it.

        Args:
            shift: How far each channel is pulled, in whole pixels, drawn per axis. This has to stay
                under a stroke width: a shift wider than the stroke pulls all three channels off it
                at once, which fades the stroke itself rather than only fringing its edge.
            mix: How much of the pulled channel is mixed in. At the highest default a stroke
                thinner than the shift still keeps three quarters of its contrast.
            gain: Factor the pulled channel is scaled by.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.shift = shift
        self.mix = mix
        self.gain = gain

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        # A distinct direction per channel, so the three do not travel together and cancel out.
        directions = torch.argsort(
            context.rand(context.batch_size, len(MISALIGNMENT_DIRECTIONS)), dim=1
        )

        return dict(
            directions=directions[:, :3],
            magnitudes=_draw_int(context, self.shift, 3, 2),
            gains=context.sample(self.gain, 3),
            mixes=context.sample(self.mix, 3),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        steps = torch.tensor(MISALIGNMENT_DIRECTIONS, device=image.device)

        for channel in range(3):
            offsets = (
                steps[parameters["directions"][:, channel]] * parameters["magnitudes"][:, channel]
            )

            rolled = shift(image[:, channel : channel + 1], offsets)
            rolled = rolled * per_element(parameters["gains"][:, channel], rolled)

            mix = per_element(parameters["mixes"][:, channel], rolled)
            image[:, channel : channel + 1].lerp_(rolled, mix)

        return image.clamp_(0, 1)


class PaperTint(PixelTransform):
    def __init__(
        self,
        hue: Parameter = (0.05, 0.18),
        strength: Parameter = (0.0, 0.12),
        p: float = 1.0,
    ) -> None:
        """Tints the paper stock, cream through yellow, leaving its brightness alone.

        The tint is a color whose largest channel is exactly one, so the lightest pixels of the page
        keep their brightness and only the other two channels come down. That is cheaper than a
        round trip through HSV by an order of magnitude on a page-sized image, and it leaves ink
        alone for the same reason it leaves paper bright.

        Args:
            hue: Hue of the tint in `[0, 1)`, the default covering cream to yellow.
            strength: How far the tint is taken.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.hue = hue
        self.strength = strength

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(hue=context.sample(self.hue), strength=context.sample(self.strength))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        # Fully saturated and at full value, so the largest of the three channels is exactly one.
        hue = parameters["hue"]
        ones = torch.ones_like(hue)
        color = hsv_to_rgb(torch.stack((hue, ones, ones), dim=1).view(-1, 3, 1, 1))

        strength = per_element(parameters["strength"], image)

        return darken(image, 1 - strength * (1 - color))


class HalftoneDither(PixelTransform):
    def __init__(
        self,
        pattern: DitherPattern = DitherPattern.CLUSTERED,
        size: Parameter = (3, 6),
        cell: Parameter = (1, 1),
        ink_range: tuple[float, float] = (0.15, 0.85),
        strength: Parameter = (0.3, 0.6),
        sigma: float = 1.0,
        p: float = 1.0,
    ) -> None:
        """Screens the page into dots, the way a fax, a photocopier or a dot matrix printer does.

        Halftoning and ordered dithering are the same operation with different threshold tiles, so
        one class covers all three looks through `DitherPattern`.

        `ink_range` is what keeps the page readable and is not a tuning knob. Tone below its low
        end goes solid black and tone above its high end goes solid white, whatever the tile says,
        so only midtones break into dots: the antialiased rim of a glyph, a gray fill, an image.
        A hard threshold over the whole page would instead drop faint glyphs and hairline rules,
        which is why binarization is not offered.

        Args:
            pattern: Which screen to use.
            size: Number of thresholds along a screen period, a whole number.
            cell: Pixels per threshold, a whole number. Together with `size` this sets the
                screen period, which has to stay under the width of a stroke and not merely under
                the height of a glyph. A form's table cells carry light text on a colored fill, and
                because the fill is a midtone it gets screened along with the text on it; past a
                period of about eight pixels a stroke falls inside a single screen cell and is
                averaged out of existence.
            ink_range: The fixed band of tones, as `(low, high)`, that is allowed to break into
                dots. Everything outside it is left solid. It is not drawn.
            strength: How far the page is taken toward the screened version. Near
                one the screen replaces the page instead of blending with it, which is the other
                way light text on a fill disappears, so this stays well below.
            sigma: Standard deviation of the blur applied before the screen, standing in for the
                optics of the machine that did the screening. Without it a crisp page has almost no
                midtone to screen, every pixel falls outside `ink_range`, and the dots barely show
                at all. This is the knob that decides how hard the screen bites, which is why it is
                not drawn from a range: at one a page reads as a fax and stays legible, and by two
                the dots have eaten enough of every glyph that it does not.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.pattern = DitherPattern(pattern)
        self.size = size
        self.cell = cell
        self.ink_range = ink_range
        self.strength = strength
        self.sigma = sigma

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        batch_size = context.batch_size

        # Every page has its own screen, the way it would have come off its own machine.
        sizes = _draw_int(context, self.size)

        # A line screen runs either way across the page.
        transposed = context.rand(batch_size) < 0.5
        if self.pattern is not DitherPattern.LINE:
            transposed = torch.zeros_like(transposed)

        offsets = torch.floor(context.rand(batch_size, 2) * (4 * sizes[:, None] + 1))

        return dict(
            patterns=torch.full((batch_size,), DITHER_PATTERNS.index(self.pattern)),
            sizes=sizes,
            cells=_draw_int(context, self.cell),
            transposed=transposed,
            offsets=offsets.to(dtype=torch.int64),
            strength=context.sample(self.strength),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        threshold = _threshold_maps(
            parameters["patterns"],
            parameters["sizes"],
            parameters["cells"],
            parameters["transposed"],
            parameters["offsets"],
            image.shape[2:],
            image.device,
            image.dtype,
        )

        low, high = self.ink_range

        # Softened first, standing in for the optics of the screening machine. This is what gives
        # the screen a midtone to work with on an otherwise crisp page.
        gray = gaussian_blur(rgb_to_grayscale(image), self.sigma)
        gray = ((gray - low) / (high - low)).clamp_(0, 1)

        screened = (gray > threshold).to(dtype=image.dtype)

        return image.lerp_(screened, per_element(parameters["strength"], image))


class LowInkStreaks(PixelTransform):
    def __init__(
        self,
        count: Parameter = (2, 8),
        alpha: Parameter = (0.10, 0.45),
        coverage: Parameter = (0.5, 0.9),
        periodic_probability: float = 0.35,
        period: Parameter = (24, 64),
        vertical_probability: float = 0.3,
        dash_size: int = 16,
        p: float = 1.0,
    ) -> None:
        """Draws pale streaks straight through the page, a printer running out of ink.

        The streaks only lighten, so they can dim what they cross but never replace it. That
        matters most for a hairline rule, which is one or two pixels tall and runs the same way a
        horizontal streak does: an opaque streak landing on one would erase it outright while its
        label still says it is there. `alpha` is what forbids that, and it is capped low
        enough that a rule survives even a streak stacked on top of `LineFragmentation`.

        Args:
            count: How many streaks are drawn when they are placed at random, a whole number.
            alpha: How far a streak lightens what it crosses, drawn per streak.
            coverage: How much of a streak's length is actually pale.
            periodic_probability: Probability the streaks are evenly spaced rather than random.
            period: Spacing between evenly spaced streaks, in whole pixels.
            vertical_probability: Probability the streaks run down the page instead of across.
            dash_size: Length scale of the gaps along a streak, in pixels.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.count = count
        self.alpha = alpha
        self.coverage = coverage
        self.periodic_probability = periodic_probability
        self.period = period
        self.vertical_probability = vertical_probability
        self.dash_size = dash_size

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        batch_size = context.batch_size
        height, width = context.shape[-2:]
        length = max(height, width)

        # Built as streaks running along the width, and transposed for the vertical variant.
        vertical = context.rand(batch_size) < self.vertical_probability
        across = torch.where(vertical, width, height)
        along = torch.where(vertical, height, width)
        positions = torch.arange(length)[None]

        # Evenly spaced streaks. The phase is kept on the page, which it would otherwise run past
        # whenever the period is wider than the page is.
        periods = _draw_int(context, self.period)
        phases = torch.floor(context.rand(batch_size) * torch.minimum(periods, across))[:, None]
        even = (positions >= phases) & (torch.remainder(positions - phases, periods[:, None]) == 0)
        even = even & (positions < across[:, None])
        even_alpha = torch.where(even, context.sample(self.alpha, length), 0.0)

        # Streaks at random.
        most = _highest(self.count)
        counts = _draw_int(context, self.count)
        places = torch.floor(context.rand(batch_size, most) * across[:, None]).to(dtype=torch.int64)
        drawn = context.sample(self.alpha, most) * (torch.arange(most)[None] < counts[:, None])
        random_alpha = torch.zeros(batch_size, length, dtype=torch.float64)
        random_alpha.scatter_reduce_(1, places, drawn, reduce="amax")

        periodic = context.rand(batch_size) < self.periodic_probability
        alpha = torch.where(periodic[:, None], even_alpha, random_alpha)

        # Feather onto the neighboring lines, so a streak is not one hard row of pixels.
        neighbors = F.max_pool1d(alpha[:, None], 3, stride=1, padding=1)[:, 0]
        alpha = torch.maximum(alpha, 0.35 * neighbors)

        # The gaps come in runs, the way ink actually fails, rather than pixel by pixel.
        coarse = context.rand(batch_size, max(2, length // self.dash_size))
        coverage = context.sample(self.coverage)
        dash = torch.zeros(batch_size, length, dtype=torch.float64)

        for size in torch.unique(along).tolist():
            indices = torch.nonzero(along == size)[:, 0]
            cells = max(2, size // self.dash_size)

            profile = F.interpolate(
                coarse[indices, None, :cells], size=size, mode="linear", align_corners=False
            )[:, 0]
            dash[indices, :size] = (profile < coverage[indices, None]).to(dtype=torch.float64)

        return dict(vertical=vertical, alpha=alpha, dash=dash)

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=False)

        height, width = image.shape[2:]
        vertical = parameters["vertical"]
        alpha, dash = parameters["alpha"], parameters["dash"]

        field = torch.empty(len(image), 1, height, width, device=image.device, dtype=image.dtype)

        horizontal = ~vertical
        if bool(horizontal.any()):
            field[horizontal] = (alpha[horizontal, :height, None] * dash[horizontal, None, :width])[
                :, None
            ]

        if bool(vertical.any()):
            field[vertical] = (dash[vertical, :height, None] * alpha[vertical, None, :width])[
                :, None
            ]

        return lighten(image, field)


class RollerStreaks(PixelTransform):
    def __init__(
        self,
        bar_width: Parameter = (8, 12),
        value_range: tuple[float, float] = (0.62, 0.99),
        strength: Parameter = (0.0, 0.25),
        envelope: Parameter = (10, 25),
        vertical_probability: float = 0.5,
        p: float = 1.0,
    ) -> None:
        """Draws the soft streaks a scanner's dirty transport rollers leave across a page.

        The streaks are a single one-dimensional profile broadcast over the page, so this costs one
        multiply per pixel and no temporary the size of the page at all.

        Args:
            bar_width: Pixels in half a bar, a whole number.
            value_range: The fixed band, as factors on the page, that the high and low values of
                every bar are drawn within. It is not drawn itself.
            strength: How far the profile is taken. Zero leaves the page untouched.
            envelope: How many times wider the slow envelope is than the bars, a whole number.
            vertical_probability: Probability the streaks run down the page instead of across.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        if _lowest(bar_width) < 1 or _lowest(envelope) < 1:
            raise ValueError("Bars and envelopes must be at least one pixel wide.")

        self.bar_width = bar_width
        self.value_range = value_range
        self.strength = strength
        self.envelope = envelope
        self.vertical_probability = vertical_probability

    def _draw_profiles(
        self, context: Context, length: int, widths: torch.Tensor, value_range: tuple[float, float]
    ) -> torch.Tensor:
        """Draws bar profiles, every bar with its own pair of values, so streaks vary in depth."""

        count = -(-length // (2 * int(widths.min())))

        highs = context.sample(value_range, count)
        lows = value_range[0] + (highs - value_range[0]) * context.rand(context.batch_size, count)

        return bar_profiles(length, widths, highs, lows)

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        batch_size = context.batch_size
        length = max(context.shape[-2:])

        vertical = context.rand(batch_size) < self.vertical_probability
        widths = _draw_int(context, self.bar_width)
        envelope_widths = widths * _draw_int(context, self.envelope)

        # The bars themselves, and a much slower profile that decides where they bite at all.
        profile = self._draw_profiles(context, length, widths, self.value_range)
        envelope = self._draw_profiles(context, length, envelope_widths, (0.9, 1.0))

        strength = context.sample(self.strength)[:, None]
        field = 1 - strength * (1 - profile * envelope)

        return dict(vertical=vertical, field=field)

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=False)

        height, width = image.shape[2:]
        vertical = parameters["vertical"][:, None, None, None]
        profile = parameters["field"]

        down, across = profile[:, None, :height, None], profile[:, None, None, :width]

        # A batch that streaks one way is darkened by the profile alone, broadcast over the page.
        if bool(vertical.all()):
            return darken(image, down)

        if not bool(vertical.any()):
            return darken(image, across)

        return darken(image, torch.where(vertical, down, across))


class InkSpread(PixelTransform):
    def __init__(
        self,
        rate: Parameter = (0.3, 0.6),
        strength: Parameter = (0.0, 0.5),
        cell_size: int = 3,
        sigma: float = 0.8,
        p: float = 1.0,
    ) -> None:
        """Fattens and darkens ink where it meets paper, the way it soaks into fibers.

        Only the rim of paper immediately outside a stroke is touched, and only by one pixel, so no
        counter can close and no thin stroke can thicken into a blob.

        Args:
            rate: How much of the rim takes ink.
            strength: How dark the rim goes. Zero leaves the page untouched.
            cell_size: Size of a clump of soaked pixels, in pixels.
            sigma: Standard deviation of the blur softening the result.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.rate = rate
        self.strength = strength
        self.cell_size = cell_size
        self.sigma = sigma

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(rate=context.sample(self.rate), strength=context.sample(self.strength))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        gray = rgb_to_grayscale(image)
        noise = value_noise(
            gray.shape,
            self.cell_size,
            context.device_generator(image.device),
            image.device,
            image.dtype,
        )

        # Eroding grows the ink, so the difference is exactly the rim of paper it grows into.
        delta = gray - erode(gray, INK_MORPHOLOGY_SIZE)
        delta *= noise < per_element(parameters["rate"], gray)

        delta = gaussian_blur(delta, self.sigma)

        return darken(image, delta.mul_(-per_element(parameters["strength"], delta)).add_(1))


class InkErosion(PixelTransform):
    def __init__(
        self,
        coverage: Parameter = (0.1, 0.35),
        strength: Parameter = (0.0, 0.4),
        cell_size: int = 6,
        p: float = 1.0,
    ) -> None:
        """Eats speckled holes into the ink, the letterpress or worn photocopy look.

        Only the outer layer of a stroke is touched, and only by one pixel, so a three pixel stroke
        always keeps a one pixel core and stays legible. The holes are clumped rather than drawn
        per pixel, so what survives is contiguous ink instead of a random scatter of it.

        Args:
            coverage: How much of the outer layer is eaten away.
            strength: How far the eaten pixels go toward paper. Zero leaves the page
                untouched.
            cell_size: Size of a clump of eaten pixels, in pixels.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.coverage = coverage
        self.strength = strength
        self.cell_size = cell_size

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(
            coverage=context.sample(self.coverage),
            strength=context.sample(self.strength),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        gray = rgb_to_grayscale(image)
        noise = value_noise(
            gray.shape,
            self.cell_size,
            context.device_generator(image.device),
            image.device,
            image.dtype,
        )

        # Dilating shrinks the ink, so the difference is exactly the outer layer of every stroke.
        delta = dilate(gray, INK_MORPHOLOGY_SIZE) - gray
        delta *= noise < per_element(parameters["coverage"], gray)

        return lighten(image, delta.mul_(per_element(parameters["strength"], delta)))


class LineFragmentation(PixelTransform):
    def __init__(
        self,
        rate: Parameter = (0.05, 0.25),
        strength: Parameter = (0.0, 0.5),
        run_length: int = 7,
        cell_size: int = 3,
        downscale: int = 4,
        p: float = 1.0,
    ) -> None:
        """Makes long rules and underlines flake and go patchy.

        Only strokes that run unbroken for a long way in one direction are touched, which is what
        table rules and underlines do and what a glyph never does. The work is done at a quarter
        scale so a missing piece is a nibble out of the rule rather than pixel noise on it.

        The rule is faded, never erased: `strength` stops short of one on purpose. A rule is
        something a label such as a box claims on its own, with no other evidence for it on the
        page, so a rule lightened away would teach a model to invent rules out of blank paper.
        Fading one instead leaves something that is unmistakably still a rule.

        Args:
            rate: How much of a rule flakes away.
            strength: How far the flaked pieces go toward paper. Must stay below one.
            run_length: Length a stroke has to run to count as a rule, at the downscaled resolution
                and in pixels. Must be odd.
            cell_size: Size of a flake, at the downscaled resolution and in pixels.
            downscale: Factor the rules are found and flaked at.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        if to_distribution(strength).high >= 1:
            raise ValueError("Strength must stay below one, so a rule is never erased.")

        self.rate = rate
        self.strength = strength
        self.run_length = run_length
        self.cell_size = cell_size
        self.downscale = downscale

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        return dict(rate=context.sample(self.rate), strength=context.sample(self.strength))

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        height, width = image.shape[2:]

        gray = rgb_to_grayscale(image)
        dark = 1 - F.avg_pool2d(gray, self.downscale, ceil_mode=True)

        # What survives an opening along either axis is what runs far enough to be a rule.
        runs = torch.maximum(
            open_runs(dark, self.run_length, horizontal=True),
            open_runs(dark, self.run_length, horizontal=False),
        )

        noise = value_noise(
            runs.shape,
            self.cell_size,
            context.device_generator(image.device),
            image.device,
            image.dtype,
        )
        runs *= noise < per_element(parameters["rate"], runs)

        runs = F.interpolate(runs, size=(height, width), mode="bilinear", align_corners=False)

        return lighten(image, runs.mul_(per_element(parameters["strength"], runs)))


class EdgeShadow(PixelTransform):
    def __init__(
        self,
        count: Parameter = (1, 2),
        width: Parameter = (0.01, 0.06),
        strength: Parameter = (0.1, 0.4),
        p: float = 1.0,
    ) -> None:
        """Darkens a band along an edge of the image, where a sheet lifts off the scanner glass.

        This is only the edge of the sheet when the sheet fills the image. A page that has been
        padded onto a larger canvas has its own edge somewhere inside, and the band then reads as
        the edge of the scanner bed instead.

        Args:
            count: How many edges get a band, a whole number.
            width: Width of the band, as a fraction of the side it lies across.
            strength: How dark the band goes at the very edge.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.count = count
        self.width = width
        self.strength = strength

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        batch_size = context.batch_size
        height, width = context.shape[-2:]

        counts = _draw_int(context, self.count)

        # A band along the top or bottom edge is a profile down the page, one along the left or
        # right edge a profile across it. Bands multiply, so each direction is one profile.
        down = torch.ones(batch_size, height, dtype=torch.float64)
        across = torch.ones(batch_size, width, dtype=torch.float64)

        for band in range(_highest(self.count)):
            active = band < counts
            vertical = context.rand(batch_size) < 0.5
            fraction = context.sample(self.width)
            strength = context.sample(self.strength)[:, None]
            at_start = context.rand(batch_size) < 0.5

            for profile, length, selected in ((down, height, vertical), (across, width, ~vertical)):
                positions = torch.arange(length)[None]
                distances = torch.where(at_start[:, None], positions, length - 1 - positions)

                widths = torch.floor(fraction * length).clamp(min=1)[:, None]
                ramp = 1 - strength + strength * distances / (widths - 1).clamp(min=1)
                ramp = torch.where(distances < widths, ramp, 1.0)

                profile *= torch.where((active & selected)[:, None], ramp, 1.0)

        return dict(down=down, across=across)

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=False)

        field = parameters["down"][:, None, :, None] * parameters["across"][:, None, None, :]

        return darken(image, field)


class BitonalCopy(PixelTransform):
    def __init__(
        self,
        spread: Parameter = (0.4, 1.0),
        gamma: Parameter = (2.0, 3.5),
        threshold: Parameter = (0.62, 0.8),
        stain_edges: Parameter = (0, 2),
        stain_reach: Parameter = (0.1, 0.4),
        stain_density: Parameter = (0.15, 0.5),
        blotch_density: Parameter = (0.0, 0.15),
        stray_density: Parameter = (0.0, 0.004),
        grain: Parameter = (1, 3),
        streak: Parameter = (1, 3),
        noise_std: Parameter = (0.02, 0.06),
        text_reach: int = 12,
        text_density: float = 0.12,
        screen_size: Parameter = (3, 6),
        pictures: str | None = None,
        p: float = 1.0,
    ) -> None:
        """Turns the page to pure black and white, the way a fax or an old photocopier does.

        A bitonal copy keeps no grays at all: every pixel comes out black or white, by whether the
        original was darker than a threshold where it stood. Three things follow from that, and each
        is made here:

        - The strokes come out bold and ragged. The copier's optics spread the ink before it is
          thresholded, so a stroke grows by however much of the spread is still darker than the
          threshold, and its edge frays with the noise.
        - A stained, grayed or textured original, or a dirty platen, comes out as speckle: where
          the paper was near the threshold, its grain and fibers fall either side of it, as
          clusters of black specks, denser the darker the stain. The staining creeps in from an
          edge or two, where the sheet was handled or lifted off the glass, and lies in blotches
          besides, and a few stray specks lie anywhere. However stained, never much more than half
          of the paper comes out black, so the text stays legible through it, as it does on a real
          copy, and the labels still say what can be read.
        - Anything lighter than the threshold is gone. So the page's tones are first bent toward
          black, which a copier's darkness setting does too, and text the page draws in a light
          color comes through as black rather than not at all; only the paper stays white.

        Two things are kept from it, so that what the labels say can still be read. A picture is
        screened rather than thresholded, as a copier's photo mode does: it comes out black and
        white too, but as dots finer than a stroke, which keep its tones, so it still reads as a
        picture and the writing in it stays legible where a threshold would have turned it into
        blots. The speckle lies over it as over the rest. Where the pictures are is read from a
        mask (see `pictures`), and a page without one has none. And around the text the speckle is
        kept light, however stained the paper, so that no blotch runs a word into itself; the
        stains lie heavy only in the margins.

        The page is measured against its own paper where it lies before any of it, as a copier's
        exposure does, so that a page the lamp lit unevenly or that is tinted does not go black in
        patches.

        Args:
            spread: Standard deviation of the spread, in pixels.
            gamma: The gamma the page's tones are bent toward black by.
            threshold: The threshold, as a gray level after the page is stretched to
                run from black to white.
            stain_edges: How many edges the staining creeps in from, a whole number.
            stain_reach: How far in from an edge the staining reaches, as a share of
                the side it lies across.
            stain_density: How much of the paper comes out black at a stained edge.
            blotch_density: How much of it comes out black at the heart of a blotch.
            stray_density: How much of it comes out black anywhere, as stray specks.
            grain: Size of the paper's grain, in whole pixels.
            streak: How many times longer the grain is one way than the other, a whole number, the
                way paper fibers lie.
            noise_std: Standard deviation of the noise on the original.
            text_reach: How far from ink the speckle is kept light, in pixels.
            text_density: The most of the paper that comes out black within that reach.
            screen_size: Period of the screen a picture is copied with, in whole
                pixels. It stays at or below six, or a stroke in a picture falls inside a single
                dot of it and is gone (see `HalftoneDither`).
            pictures: Name of a mask target that is one on pictures and zero elsewhere, in its
                first channel. Without it, the page has no pictures.
            p: Probability of applying the augmentation per batch element.
        """

        super().__init__(p)

        self.spread = spread
        self.gamma = gamma
        self.threshold = threshold
        self.stain_edges = stain_edges
        self.stain_reach = stain_reach
        self.stain_density = stain_density
        self.blotch_density = blotch_density
        self.stray_density = stray_density
        self.grain = grain
        self.streak = streak
        self.noise_std = noise_std
        self.text_reach = text_reach
        self.text_density = text_density
        self.screen_size = screen_size
        self.pictures = pictures

    def get_parameters(self, context: Context) -> dict[str, torch.Tensor]:
        batch_size = context.batch_size
        height, width = context.shape[-2:]

        # The staining that creeps in from edges, as the strongest profile down the page and the
        # strongest across it: heaviest at the edge, and gone well before the reach runs out.
        counts = _draw_int(context, self.stain_edges)
        down = torch.zeros(batch_size, height, dtype=torch.float64)
        across = torch.zeros(batch_size, width, dtype=torch.float64)

        for edge in range(_highest(self.stain_edges)):
            active = edge < counts
            vertical = context.rand(batch_size) < 0.5
            reach_fraction = context.sample(self.stain_reach)
            flipped = context.rand(batch_size) < 0.5
            density = context.sample(self.stain_density)[:, None]

            for profile, length, selected in ((across, width, vertical), (down, height, ~vertical)):
                positions = torch.arange(length, dtype=torch.float64)[None]
                distances = torch.where(flipped[:, None], length - 1 - positions, positions)
                reach = (reach_fraction * length).clamp(min=1)[:, None]

                stain = (1 - distances / reach).clamp(min=0).pow(1.5) * density
                profile.copy_(
                    torch.where(
                        (active & selected)[:, None], torch.maximum(profile, stain), profile
                    )
                )

        # The grain of the paper: clumps drawn out one way, like fibers.
        grains = _draw_int(context, self.grain)
        streaks = _draw_int(context, self.streak)
        along_height = context.rand(batch_size) < 0.5
        cell_heights = torch.where(along_height, grains * streaks, grains)
        cell_widths = torch.where(along_height, grains, grains * streaks)

        # The screen a picture is copied with.
        screen_sizes = _draw_int(context, self.screen_size)
        screen_patterns = torch.where(
            context.rand(batch_size) < 0.5,
            DITHER_PATTERNS.index(DitherPattern.BAYER),
            DITHER_PATTERNS.index(DitherPattern.CLUSTERED),
        )
        screen_offsets = torch.floor(context.rand(batch_size, 2) * (4 * screen_sizes[:, None] + 1))

        return dict(
            gamma=context.sample(self.gamma),
            spread=context.sample(self.spread),
            noise_std=context.sample(self.noise_std),
            threshold=context.sample(self.threshold),
            stray=context.sample(self.stray_density),
            blotch=context.sample(self.blotch_density),
            stain_down=down,
            stain_across=across,
            cell_heights=cell_heights,
            cell_widths=cell_widths,
            screen_patterns=screen_patterns,
            screen_sizes=screen_sizes,
            screen_offsets=screen_offsets.to(dtype=torch.int64),
        )

    def apply_image(
        self, image: torch.Tensor, parameters: dict[str, torch.Tensor], context: Context
    ) -> torch.Tensor:
        _check(self, image, rgb=True)

        batch_size, channels, height, width = image.shape
        device, dtype = image.device, image.dtype
        generator = context.device_generator(device)

        on_picture = torch.zeros(batch_size, 1, height, width, device=device, dtype=torch.bool)
        if self.pictures is not None:
            pictures = context.raster(self.pictures)
            if pictures is None:
                raise ValueError(f"There is no mask target {self.pictures!r} with the pictures.")

            on_picture = pictures[:, :1] > 0.5

        gray = self._expose(rgb_to_grayscale(image))

        copied = gray.pow(per_element(parameters["gamma"], gray))
        copied = gaussian_blur(copied, parameters["spread"])
        noise = torch.randn(copied.shape, generator=generator, device=device, dtype=dtype)
        copied = copied + noise * per_element(parameters["noise_std"], copied)

        ink = copied < per_element(parameters["threshold"], copied)

        # Where the paper is stained, its grain decides which of it comes out black, but never
        # much of it near the text.
        density = self._get_density(
            parameters, (batch_size, 1, height, width), generator, device, dtype
        )

        # Fading in over the reach rather than at once, so the text does not stand in a halo.
        near_text = dilate(ink.to(dtype), 2 * self.text_reach + 1)
        near_text = gaussian_blur(near_text, self.text_reach / 2).clamp(0, 1)

        density = near_text * density.clamp(max=self.text_density) + (1 - near_text) * density

        speckle = (
            self._get_grain(parameters, (batch_size, 1, height, width), generator, device, dtype)
            < density
        )

        # A picture is screened instead of thresholded, and the speckle lies over it all the same.
        if bool(on_picture.any()):
            ink = torch.where(on_picture, self._screen(gray, parameters), ink)

        copy = 1 - (ink | speckle).to(dtype)

        return image.copy_(copy.expand(batch_size, channels, height, width))

    def _screen(self, gray: torch.Tensor, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns where a screen of dots turns the exposed page black, as photo mode copies it."""

        count = len(gray)

        threshold = _threshold_maps(
            parameters["screen_patterns"],
            parameters["screen_sizes"],
            torch.ones(count, dtype=torch.int64),
            torch.zeros(count, dtype=torch.bool),
            parameters["screen_offsets"],
            gray.shape[2:],
            gray.device,
            gray.dtype,
        )

        # Softened a little first, as the copier's optics do, so the screen has tones to work with.
        return gaussian_blur(gray, 0.8) < threshold

    def _expose(self, gray: torch.Tensor) -> torch.Tensor:
        """Returns the page measured against its own paper where it lies, as a copier exposes it.

        The paper is the lightest the page gets around each pixel, so that paper that the lamp lit
        unevenly or that is tinted comes out white, and only what is darker than it shows.
        """

        height, width = gray.shape[2:]

        # The lightest, at a quarter of the size and over a few text heights, softly.
        paper = dilate(F.avg_pool2d(gray, PAPER_SCALE, ceil_mode=True), PAPER_WINDOW)
        paper = F.interpolate(paper, size=(height, width), mode="bilinear", align_corners=False)
        paper = gaussian_blur(paper, 8.0)

        low = gray.amin(dim=(1, 2, 3), keepdim=True)

        return ((gray - low) / (paper - low).clamp(min=0.1)).clamp(0, 1)

    def _get_density(
        self,
        parameters: dict[str, torch.Tensor],
        shape: tuple[int, int, int, int],
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns how much of the paper comes out black at each pixel, by how stained it was."""

        batch_size, _, height, width = shape

        stain = torch.maximum(
            parameters["stain_down"][:, None, :, None], parameters["stain_across"][:, None, None, :]
        )
        stain = torch.maximum(stain, per_element(parameters["stray"], stain))

        # Blotches, where the original was marked or the platen dirty.
        blotches = torch.randn(batch_size, 1, 4, 3, generator=generator, device=device, dtype=dtype)
        blotches = F.interpolate(
            blotches, size=(height, width), mode="bicubic", align_corners=False
        )
        stain = stain + blotches.clamp(min=0, max=1) * per_element(parameters["blotch"], stain)

        return stain.clamp(max=MAX_SPECKLE_DENSITY)

    def _get_grain(
        self,
        parameters: dict[str, torch.Tensor],
        shape: tuple[int, int, int, int],
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns the paper's grain, in `[0, 1)`: clumps drawn out one way, like fibers."""

        batch_size, _, height, width = shape
        grain = torch.empty(shape, device=device, dtype=dtype)

        keys = torch.stack((parameters["cell_heights"], parameters["cell_widths"]), dim=1).cpu()
        for key, indices in _groups(keys):
            cell_height, cell_width = (int(x) for x in key)

            noise = torch.rand(
                len(indices),
                1,
                max(1, -(-height // cell_height)),
                max(1, -(-width // cell_width)),
                generator=generator,
                device=device,
                dtype=dtype,
            )

            grain[indices.to(device)] = F.interpolate(
                noise, size=(height, width), mode="bilinear", align_corners=False
            )

        return grain
