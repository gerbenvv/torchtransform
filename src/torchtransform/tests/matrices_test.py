from unittest import TestCase

import torch

from torchtransform import matrices as mat


class MatricesTest(TestCase):
    def test_rotations_turn_x_toward_y(self) -> None:
        angles = torch.tensor((90.0,), dtype=torch.float64)
        point = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)

        torch.testing.assert_close(
            mat.apply(mat.rotation_2d(angles), point),
            torch.tensor([[[0.0, 1.0]]], dtype=torch.float64),
        )

        axes = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        point = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
        torch.testing.assert_close(
            mat.apply(mat.rotation_3d(axes, angles), point),
            torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64),
        )

    def test_quaternion_rotations_are_orthonormal(self) -> None:
        rotations = mat.rotation_from_quaternions(torch.randn(16, 4, dtype=torch.float64))[
            :, :3, :3
        ]

        torch.testing.assert_close(
            rotations @ rotations.transpose(1, 2),
            torch.eye(3, dtype=torch.float64).expand(16, -1, -1),
        )
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(16, dtype=torch.float64))

    def test_homography_maps_the_corners(self) -> None:
        sources = torch.tensor(
            [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]], dtype=torch.float64
        )
        targets = sources + torch.randn(1, 4, 2, dtype=torch.float64) * 0.2

        matrices = mat.homography(sources, targets)

        self.assertTrue(mat.is_projective(matrices))
        torch.testing.assert_close(mat.apply(matrices, sources), targets)

    def test_signed_permutations(self) -> None:
        turn = mat.rotation_2d(torch.tensor((90.0, 30.0), dtype=torch.float64))
        permutations, signs, exact = mat.signed_permutations(turn)

        self.assertEqual(exact.tolist(), [True, False])
        self.assertEqual(permutations[0].tolist(), [1, 0])
        self.assertEqual(signs[0].tolist(), [-1.0, 1.0])

    def test_shear(self) -> None:
        coefficients = torch.zeros(1, 2, 2, dtype=torch.float64)
        coefficients[0, 0, 1] = 0.5

        point = torch.tensor([[[0.0, 2.0]]], dtype=torch.float64)
        torch.testing.assert_close(
            mat.apply(mat.shear(coefficients), point),
            torch.tensor([[[1.0, 2.0]]], dtype=torch.float64),
        )
