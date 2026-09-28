"""Helpers shared by the tests."""

import torch

# Devices to test on.
DEVICES: tuple[torch.device, ...] = (torch.device("cpu"),) + (
    (torch.device("cuda"),) if torch.cuda.is_available() else ()
)


def blobs(points: torch.Tensor, shape: tuple[int, ...], sigma: float = 1.5) -> torch.Tensor:
    """Returns images with a Gaussian blob at every point.

    Args:
        points: Points of shape `(B, n)` in pixel coordinates.
        shape: Spatial shape of the images.
        sigma: Standard deviation of the blobs in pixels.

    Returns:
        Images of shape `(B, 1, *shape)`.
    """

    grids = torch.meshgrid(
        *(torch.arange(s, dtype=torch.float64) + 0.5 for s in shape), indexing="ij"
    )
    centers = torch.stack(grids[::-1], dim=-1)

    distances = ((centers[None] - points.view(len(points), *(1,) * len(shape), -1)) ** 2).sum(
        dim=-1
    )

    return torch.exp(-distances / (2 * sigma**2))[:, None].to(dtype=torch.float32)


def centroids(images: torch.Tensor) -> torch.Tensor:
    """Returns the weighted centroid of every image of shape `(B, 1, ...)`, of shape `(B, n)`."""

    shape = images.shape[2:]
    grids = torch.meshgrid(
        *(torch.arange(s, dtype=torch.float64) + 0.5 for s in shape), indexing="ij"
    )
    centers = torch.stack(grids[::-1], dim=-1)

    weights = images[:, 0].to(dtype=torch.float64).cpu()
    total = weights.flatten(1).sum(dim=1)

    return (weights[..., None] * centers[None]).flatten(1, -2).sum(dim=1) / total[:, None]
