"""
Embedded rigid / rigid+scale / rigid+scale+shear → affine helpers.

The decoder learns per-subject ``embed`` parameters of dimension 6, 9 or 12:

- ``embed[:3]`` — Euler angles around x, y, z (in radians).
- ``embed[3:6]`` — translation (in normalised coords, i.e. fractions of the
  world bounding-box half-extent — same units as ``coords`` themselves).
- ``embed[6:9]`` — additive scale offsets (1 + s applied per axis).
- ``embed[9:12]`` — shear offsets (off-diagonal in 3 shear matrices).
"""

from __future__ import annotations

import torch


def euler2rot(theta: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles ``(..., 3)`` to a rotation matrix ``(..., 3, 3)``."""
    c1, s1 = torch.cos(theta[..., 0]), torch.sin(theta[..., 0])
    c2, s2 = torch.cos(theta[..., 1]), torch.sin(theta[..., 1])
    c3, s3 = torch.cos(theta[..., 2]), torch.sin(theta[..., 2])
    r11 = c1 * c3 - c2 * s1 * s3
    r12 = -c1 * s3 - c2 * c3 * s1
    r13 = s1 * s2
    r21 = c3 * s1 + c1 * c2 * s3
    r22 = c1 * c2 * c3 - s1 * s3
    r23 = -c1 * s2
    r31 = s2 * s3
    r32 = c3 * s2
    r33 = c2
    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1)
    return R.view(R.shape[:-1] + (3, 3))


def embed2affine(embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a 6/9/12-D embedding into ``(R, t)`` per row."""
    if embed.shape[-1] not in (6, 9, 12):
        raise ValueError(
            f"embed last-dim must be 6, 9 or 12; got {embed.shape[-1]}"
        )
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:6]
    if embed.shape[-1] >= 9:
        S = torch.diag_embed(1.0 + embed[..., 6:9])
        R = torch.matmul(R, S)
    if embed.shape[-1] == 12:
        Sx = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        Sy = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        Sz = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        Sx[..., 0, 1] = embed[..., 10]
        Sx[..., 0, 2] = embed[..., 11]
        Sy[..., 1, 0] = embed[..., 9]
        Sy[..., 1, 2] = embed[..., 11]
        Sz[..., 2, 0] = embed[..., 9]
        Sz[..., 2, 1] = embed[..., 10]
        R = torch.matmul(R, Sx)
        R = torch.matmul(R, Sy)
        R = torch.matmul(R, Sz)
    return R, t


def apply_embedded_affine(
    coords: torch.Tensor,
    embed: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply the per-coord embedded affine to ``coords``.

    Both ``coords`` and ``embed`` are batched along the same first axis ``N``:
    one transform per coord. Returns ``(N, 3)``.
    """
    R, t = embed2affine(embed)
    if inverse:
        R = R.inverse()
        t = -torch.einsum("nij,nj->ni", R, t)
    return torch.einsum("nij,nj->ni", R, coords) + t
