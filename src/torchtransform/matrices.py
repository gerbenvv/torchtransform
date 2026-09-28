"""Homogeneous transformation matrices for 2D and 3D coordinates.

Coordinates are in `(x, y)` or `(x, y, z)` order, where `x` runs along the width, `y` along the
height and `z` along the depth of a tensor of shape `(..., D, H, W)`. Since `y` points down, the
system is right-handed with `z` pointing away from the viewer. All matrices are of shape
`(B, n + 1, n + 1)` for `n` spatial dimensions and map a column vector `(x, y, [z,] 1)`.
"""

import math

import torch

# Tolerance under which a matrix entry is taken to be exactly an integer.
INTEGER_TOLERANCE: float = 1e-6


def identity(batch_size: int, ndim: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Returns identity matrices of shape `(B, n + 1, n + 1)`."""

    return torch.eye(ndim + 1, dtype=dtype).expand(batch_size, -1, -1).clone()


def translation(translations: torch.Tensor) -> torch.Tensor:
    """Returns matrices that add `translations`, of shape `(B, n)`, to coordinates."""

    batch_size, ndim = translations.shape

    matrices = identity(batch_size, ndim, translations.dtype)
    matrices[:, :ndim, ndim] = translations

    return matrices


def scale(scales: torch.Tensor) -> torch.Tensor:
    """Returns matrices that multiply coordinates by `scales`, of shape `(B, n)`, per axis."""

    batch_size, ndim = scales.shape

    matrices = identity(batch_size, ndim, scales.dtype)
    matrices[:, range(ndim), range(ndim)] = scales

    return matrices


def rotation_2d(angles: torch.Tensor) -> torch.Tensor:
    """Returns 2D rotation matrices.

    Args:
        angles: Angles in degrees, of shape `(B,)`. A positive angle turns the x axis toward the y
            axis, which is clockwise on screen, since y points down.

    Returns:
        Matrices of shape `(B, 3, 3)`.
    """

    radians = torch.deg2rad(angles)
    cos, sin = torch.cos(radians), torch.sin(radians)

    matrices = identity(len(angles), 2, angles.dtype)
    matrices[:, 0, 0] = cos
    matrices[:, 0, 1] = -sin
    matrices[:, 1, 0] = sin
    matrices[:, 1, 1] = cos

    return matrices


def rotation_3d(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Returns 3D rotation matrices around axes by the right-hand rule.

    Around the z axis this is the same turn as `rotation_2d`: from the x axis toward the y axis.

    Args:
        axes: Rotation axes of shape `(B, 3)`. They are normalized.
        angles: Angles in degrees, of shape `(B,)`.

    Returns:
        Matrices of shape `(B, 4, 4)`.
    """

    axes = torch.nn.functional.normalize(axes, dim=1)
    x, y, z = axes.unbind(dim=1)

    radians = torch.deg2rad(angles)
    cos, sin = torch.cos(radians), torch.sin(radians)
    t = 1 - cos

    # Rodrigues' rotation formula, written out.
    matrices = identity(len(angles), 3, angles.dtype)
    matrices[:, 0, 0] = t * x * x + cos
    matrices[:, 0, 1] = t * x * y - z * sin
    matrices[:, 0, 2] = t * x * z + y * sin
    matrices[:, 1, 0] = t * x * y + z * sin
    matrices[:, 1, 1] = t * y * y + cos
    matrices[:, 1, 2] = t * y * z - x * sin
    matrices[:, 2, 0] = t * x * z - y * sin
    matrices[:, 2, 1] = t * y * z + x * sin
    matrices[:, 2, 2] = t * z * z + cos

    return matrices


def rotation_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Returns 3D rotation matrices from quaternions `(w, x, y, z)` of shape `(B, 4)`.

    The quaternions are normalized, so normalized Gaussian noise gives rotations drawn uniformly
    over all orientations.
    """

    quaternions = torch.nn.functional.normalize(quaternions, dim=1)
    w, x, y, z = quaternions.unbind(dim=1)

    matrices = identity(len(quaternions), 3, quaternions.dtype)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return matrices


def shear(coefficients: torch.Tensor) -> torch.Tensor:
    """Returns shear matrices.

    Args:
        coefficients: Coefficients of shape `(B, n, n)`, where entry `(i, j)` is how much
            coordinate `i` moves per unit of coordinate `j`. The diagonal is ignored.

    Returns:
        Matrices of shape `(B, n + 1, n + 1)`.
    """

    batch_size, ndim, _ = coefficients.shape

    off_diagonal = ~torch.eye(ndim, dtype=torch.bool)

    matrices = identity(batch_size, ndim, coefficients.dtype)
    matrices[:, :ndim, :ndim] += coefficients * off_diagonal

    return matrices


def homography(sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Returns the projective matrices that map four 2D points onto four others.

    Args:
        sources: Points of shape `(B, 4, 2)`.
        targets: Points of shape `(B, 4, 2)`, where the sources go.

    Returns:
        Matrices of shape `(B, 3, 3)`, normalized so their bottom-right entry is one.
    """

    batch_size = len(sources)
    x, y = sources.unbind(dim=2)
    u, v = targets.unbind(dim=2)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # The standard direct linear transform, with the bottom-right entry fixed at one. Every point
    # gives one row for u and one for v.
    rows_u = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=2)
    rows_v = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=2)

    system = torch.cat((rows_u, rows_v), dim=1)
    right_hand_side = torch.cat((u, v), dim=1)

    solution = torch.linalg.solve(system, right_hand_side)

    matrices = torch.ones(batch_size, 9, dtype=sources.dtype)
    matrices[:, :8] = solution

    return matrices.view(batch_size, 3, 3)


def apply(matrices: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Applies homogeneous matrices to coordinates.

    Args:
        matrices: Matrices of shape `(B, n + 1, n + 1)`.
        coordinates: Coordinates of shape `(B, ..., n)`.

    Returns:
        The transformed coordinates, of the same shape. A projective matrix divides by the
        homogeneous coordinate.
    """

    batch_size = coordinates.shape[0]
    ndim = coordinates.shape[-1]

    matrices = matrices.to(dtype=coordinates.dtype, device=coordinates.device)
    flat = coordinates.reshape(batch_size, -1, ndim)

    linear = matrices[:, None, :ndim, :ndim]
    offsets = matrices[:, None, :ndim, ndim]

    transformed = (linear @ flat[..., None])[..., 0] + offsets

    if is_projective(matrices):
        denominators = (matrices[:, None, ndim, :ndim] * flat).sum(dim=-1) + matrices[
            :, None, ndim, ndim
        ]
        transformed = transformed / denominators[..., None]

    return transformed.view(coordinates.shape)


def is_projective(matrices: torch.Tensor) -> bool:
    """Returns whether any of the matrices has a bottom row other than `(0, ..., 0, 1)`."""

    return bool(projective_elements(matrices).any())


def centered_to_pixel(
    matrices: torch.Tensor, input_size: torch.Tensor, output_size: torch.Tensor
) -> torch.Tensor:
    """Converts matrices between centered coordinates to matrices between pixel coordinates.

    Centered coordinates have their origin in the center of the canvas, pixel coordinates in the
    corner of its first pixel. Both are in pixels.

    Args:
        matrices: Matrices of shape `(B, n + 1, n + 1)` from the input to the output canvas.
        input_size: Size of the input canvas in `(x, y, [z])` order, of shape `(n,)`.
        output_size: Size of the output canvas, of shape `(n,)`.

    Returns:
        The matrices in pixel coordinates.
    """

    batch_size = len(matrices)

    from_input = translation(-0.5 * input_size.to(matrices.dtype).expand(batch_size, -1))
    to_output = translation(0.5 * output_size.to(matrices.dtype).expand(batch_size, -1))

    return to_output @ matrices @ from_input


def signed_permutations(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes the linear parts of affine matrices into signed permutations, where they are.

    Args:
        matrices: Matrices of shape `(B, n + 1, n + 1)`.

    Returns:
        The permutations, of shape `(B, n)`, where entry `i` is the column of the largest entry in
        row `i`; the signs of those entries, of shape `(B, n)`; and which matrices are affine with
        a linear part that is a signed permutation matrix, of shape `(B,)`.
    """

    ndim = matrices.shape[-1] - 1
    linear = matrices[:, :ndim, :ndim]

    rounded = torch.round(linear)
    integer = (torch.abs(linear - rounded) < INTEGER_TOLERANCE).flatten(1).all(dim=1)

    absolute = torch.abs(rounded)
    permutation = (absolute.sum(dim=1) == 1).all(dim=1) & (absolute.sum(dim=2) == 1).all(dim=1)

    permutations = torch.argmax(torch.abs(linear), dim=2)
    signs = torch.sign(torch.gather(linear, 2, permutations[..., None])[..., 0])

    return permutations, signs, integer & permutation & ~projective_elements(matrices)


def projective_elements(matrices: torch.Tensor) -> torch.Tensor:
    """Returns which matrices have a bottom row other than `(0, ..., 0, 1)`, of shape `(B,)`."""

    ndim = matrices.shape[-1] - 1

    bottom = matrices[:, ndim]
    expected = torch.zeros_like(bottom[0])
    expected[ndim] = 1

    return (torch.abs(bottom - expected) >= 1e-12).any(dim=1)


def degrees_to_shear(angles: torch.Tensor) -> torch.Tensor:
    """Converts shear angles in degrees to shear coefficients."""

    return torch.tan(angles * (math.pi / 180))
