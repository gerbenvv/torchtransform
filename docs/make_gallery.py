"""Draws the gallery images of the README.

Run with `python docs/make_gallery.py` (it needs Pillow). The images are drawn from scratch, so
the gallery needs no photographs: a scene with keypoints and a box, and a document page.
"""

import math
from pathlib import Path

import torch
from PIL import Image as PilImage, ImageDraw, ImageFont

import torchtransform as tt

# Where the images are written.
DOCS_PATH: Path = Path(__file__).parent

# Side of a tile of the gallery, and the height of its caption.
TILE_SIZE: int = 224
CAPTION_HEIGHT: int = 26

# Tiles per row.
COLUMNS: int = 6


def get_font(size: int) -> ImageFont.ImageFont:
    """Returns a font of a size, falling back to Pillow's own."""

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_scene(size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draws a scene: a sky, a sun, hills, a house and a checkered field.

    Returns:
        The image of shape `(3, size, size)` in `[0, 1]`, the corners and the top of the house as
        points of shape `(5, 2)`, and a box around the house of shape `(1, 4)`.
    """

    image = PilImage.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)

    for y in range(size):
        t = y / size
        draw.line(((0, y), (size, y)), fill=(int(90 + 100 * t), int(150 + 70 * t), 255))

    draw.ellipse((0.72 * size, 0.08 * size, 0.9 * size, 0.26 * size), fill=(255, 214, 90))
    draw.polygon(
        [
            (0, 0.62 * size),
            (0.35 * size, 0.45 * size),
            (0.7 * size, 0.6 * size),
            (size, 0.5 * size),
            (size, size),
            (0, size),
        ],
        fill=(92, 160, 80),
    )

    cell = size // 16
    for row in range(int(0.78 * size) // cell, size // cell + 1):
        for column in range(size // cell + 1):
            if (row + column) % 2 == 0:
                draw.rectangle(
                    (column * cell, row * cell, (column + 1) * cell, (row + 1) * cell),
                    fill=(70, 130, 62),
                )

    left, right, top, bottom = 0.25 * size, 0.55 * size, 0.46 * size, 0.72 * size
    roof = 0.3 * size

    draw.rectangle((left, top, right, bottom), fill=(214, 90, 70))
    draw.polygon(
        [(left - 8, top), (right + 8, top), (0.5 * (left + right), roof)], fill=(110, 60, 50)
    )
    draw.rectangle((0.36 * size, 0.58 * size, 0.44 * size, bottom), fill=(90, 60, 40))
    draw.rectangle((0.28 * size, 0.52 * size, 0.33 * size, 0.57 * size), fill=(240, 240, 200))
    draw.text(
        (0.06 * size, 0.06 * size),
        "torchtransform",
        fill=(255, 255, 255),
        font=get_font(size // 12),
    )

    pixels = (
        torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).float().view(size, size, 3)
        / 255
    )
    points = torch.tensor(
        ((left, top), (right, top), (right, bottom), (left, bottom), (0.5 * (left + right), roof))
    )
    boxes = torch.tensor(((left - 8, roof, right + 8, bottom),))

    return pixels.permute(2, 0, 1), points, boxes


def draw_page(height: int, width: int) -> torch.Tensor:
    """Draws a page with a title, text, a table and a gray picture, of shape `(3, H, W)`."""

    image = PilImage.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.text((16, 12), "Quarterly report", fill=(0, 0, 0), font=get_font(20))

    words = "the quick brown fox jumps over the lazy dog while a scanner hums".split()
    font = get_font(11)
    for line in range(9):
        text = " ".join(words[(line + i) % len(words)] for i in range(7))
        draw.text((16, 44 + 14 * line), text, fill=(20, 20, 20), font=font)

    top = 180
    for row in range(5):
        draw.line(((16, top + 18 * row), (width - 16, top + 18 * row)), fill=(0, 0, 0), width=1)
    for column in range(4):
        x = 16 + column * (width - 32) // 3
        draw.line(((x, top), (x, top + 72)), fill=(0, 0, 0), width=1)
    for row in range(4):
        for column in range(3):
            draw.text(
                (22 + column * (width - 32) // 3, top + 4 + 18 * row),
                f"{row * 3 + column + 17:>4}",
                fill=(0, 0, 0),
                font=font,
            )

    draw.rectangle((16, 262, 120, 322), fill=(150, 150, 150))
    draw.ellipse((44, 270, 92, 314), fill=(90, 90, 90))

    pixels = (
        torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        .float()
        .view(height, width, 3)
        / 255
    )

    return pixels.permute(2, 0, 1)


def to_pil(image: torch.Tensor) -> PilImage.Image:
    """Returns an image of shape `(3, H, W)` in `[0, 1]` as a Pillow image."""

    pixels = (image.clamp(0, 1) * 255).round().to(dtype=torch.uint8).permute(1, 2, 0)

    return PilImage.fromarray(pixels.numpy())


def draw_geometry(image: PilImage.Image, points: torch.Tensor, boxes: torch.Tensor) -> None:
    """Draws points and boxes onto an image."""

    draw = ImageDraw.Draw(image)

    for x0, y0, x1, y1 in boxes.tolist():
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)

    for x, y in points.tolist():
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 180), outline=(255, 255, 255))


def compose_grid(tiles: list[tuple[str, PilImage.Image]], path: Path) -> None:
    """Writes tiles with captions as a grid."""

    tile_width, tile_height = tiles[0][1].size
    rows = math.ceil(len(tiles) / COLUMNS)

    grid = PilImage.new(
        "RGB", (COLUMNS * tile_width, rows * (tile_height + CAPTION_HEIGHT)), (255, 255, 255)
    )
    draw = ImageDraw.Draw(grid)
    font = get_font(14)

    for i, (caption, tile) in enumerate(tiles):
        x = (i % COLUMNS) * tile_width
        y = (i // COLUMNS) * (tile_height + CAPTION_HEIGHT)

        grid.paste(tile, (x, y + CAPTION_HEIGHT))
        draw.text((x + 6, y + 5), caption, fill=(0, 0, 0), font=font)

    grid.save(path, optimize=True)


def make_scene_gallery() -> None:
    image, points, boxes = draw_scene(TILE_SIZE)

    gallery = (
        ("Original", tt.Identity()),
        ("Rotate", tt.Rotate((-35, -25))),
        ("Affine", tt.Affine(rotation=20, scale=(0.7, 0.8), shear=10)),
        ("Perspective", tt.Perspective((0.3, 0.3))),
        ("ElasticWarp", tt.ElasticWarp(magnitude=(6, 6), spacing=40)),
        ("LensDistortion", tt.LensDistortion((0.4, 0.4))),
        ("Twirl", tt.Twirl((80, 80), radius=(0.6, 0.6), center=(0.0, 0.0))),
        ("Wave", tt.Wave(amplitude=(5, 5), wavelength=(60, 60))),
        ("QuarterTurn", tt.QuarterTurn(turns=1)),
        ("Symmetry", tt.Symmetry().manual_seed(1)),
        ("RandomResizedCrop", tt.RandomResizedCrop(TILE_SIZE, scale=(0.3, 0.4))),
        ("AffineWithinBounds", tt.AffineWithinBounds(rotation=(-40, 40), scale=(1.5, 2.5))),
        ("ColorJitter", tt.ColorJitter(0.4, 0.4, 0.6, 0.2)),
        ("Hue", tt.Hue((0.3, 0.3))),
        ("Grayscale", tt.Grayscale()),
        ("Solarize", tt.Solarize(0.6)),
        ("Posterize", tt.Posterize(2)),
        ("Equalize", tt.Equalize()),
        ("GaussianBlur", tt.GaussianBlur((2.5, 2.5))),
        ("MotionBlur", tt.MotionBlur(length=(15, 15))),
        ("GaussianNoise", tt.GaussianNoise((0.12, 0.12))),
        ("JpegCompression", tt.JpegCompression((8, 8))),
        ("Pixelate", tt.Pixelate((6, 6))),
        ("Erasing", tt.Erasing(count=(3, 3))),
    )

    tiles = []
    for i, (caption, transform) in enumerate(gallery):
        outputs = transform(
            image=tt.Image(image[None], padding="border"),
            points=tt.Points(points[None]),
            boxes=tt.Boxes(boxes[None]),
            seed=i + 3,
        )

        tile = to_pil(outputs["image"][0])
        draw_geometry(tile, outputs["points"][0], outputs["boxes"][0])
        tiles.append((caption, tile))

    compose_grid(tiles, DOCS_PATH / "gallery.png")


def make_document_gallery() -> None:
    page = draw_page(TILE_SIZE * 3 // 2, TILE_SIZE)
    pictures = torch.zeros(1, 1, *page.shape[1:])
    pictures[..., 262:322, 16:120] = 1

    gallery = (
        ("Original", tt.Identity()),
        ("BleedThrough", tt.BleedThrough(alpha=(0.18, 0.18))),
        ("PaperTint", tt.PaperTint(strength=(0.12, 0.12))),
        ("HalftoneDither", tt.HalftoneDither(strength=(0.6, 0.6))),
        ("LowInkStreaks", tt.LowInkStreaks(alpha=(0.45, 0.45), count=(8, 8))),
        ("RollerStreaks", tt.RollerStreaks(strength=(0.25, 0.25))),
        ("InkSpread", tt.InkSpread(strength=(0.5, 0.5))),
        ("InkErosion", tt.InkErosion(strength=(0.4, 0.4), coverage=(0.35, 0.35))),
        ("LineFragmentation", tt.LineFragmentation(strength=(0.5, 0.5))),
        ("EdgeShadow", tt.EdgeShadow(count=2, width=(0.1, 0.15), strength=(0.5, 0.5))),
        ("ChannelMisalignment", tt.ChannelMisalignment(shift=(3, 3))),
        ("BitonalCopy", tt.BitonalCopy(pictures="pictures")),
    )

    tiles = []
    for i, (caption, transform) in enumerate(gallery):
        outputs = transform(image=page[None], pictures=tt.Mask(pictures), seed=i)
        tiles.append((caption, to_pil(outputs["image"][0])))

    compose_grid(tiles, DOCS_PATH / "document.png")


def make_fusion_comparison() -> None:
    image, _, _ = draw_scene(TILE_SIZE)
    images = tt.Image(image[None], padding="border")

    # Six rotations and scales that come back to where they started, applied one after the other
    # (with a no-op pixel transform in between, which forces a resampling each time) and fused.
    transforms = [tt.Rotate((20, 20)), tt.Scale((0.8, 0.8))] * 3
    transforms += [tt.Scale((1 / 0.8**3, 1 / 0.8**3)), tt.Rotate((-60, -60))]

    separate = []
    for transform in transforms:
        separate += [transform, tt.Lambda(lambda x: x)]

    tiles = [
        ("Original", to_pil(image)),
        ("One after the other", to_pil(tt.Compose(*separate)(images)[0])),
        ("Fused", to_pil(tt.Compose(*transforms)(images)[0])),
    ]

    width, height = tiles[0][1].size
    grid = PilImage.new("RGB", (3 * width, height + CAPTION_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for i, (caption, tile) in enumerate(tiles):
        grid.paste(tile, (i * width, CAPTION_HEIGHT))
        draw.text((i * width + 6, 5), caption, fill=(0, 0, 0), font=get_font(14))

    grid.save(DOCS_PATH / "fusion.png", optimize=True)


def main() -> None:
    torch.manual_seed(0)

    make_scene_gallery()
    make_document_gallery()
    make_fusion_comparison()


if __name__ == "__main__":
    main()
