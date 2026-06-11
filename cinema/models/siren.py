"""SIREN MLP with optional per-layer FiLM-style modulation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """One SIREN layer: ``sin(omega * (W x + b))``, optionally modulated.

    The forward signature is ``(coords, latent)`` and it returns
    ``(activations, latent)`` so a stack of layers can be chained with
    ``nn.Sequential`` while routing the modulation tensor through.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        *,
        is_first: bool = False,
        omega: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_lats = (
            nn.Linear(latent_features, out_features * 2, bias=bias)
            if latent_features > 0
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]):
        coords, latent = x
        h = self.linear(coords)
        if self.linear_lats is not None:
            lats = self.linear_lats(latent)
            scale = lats[..., : self.out_features]
            shift = lats[..., self.out_features :]
            out = torch.sin(self.omega * h * scale + shift)
        else:
            out = torch.sin(self.omega * h)
        return out, latent


class Siren(nn.Module):
    """Stack of ``SineLayer``s with an outermost linear head.

    ``modulated_layers`` lists the (0-indexed) hidden layers that should
    receive FiLM modulation from the latent. The outermost linear head always
    sees the latent concatenated to the last hidden activations.
    """

    def __init__(
        self,
        in_size: int,
        latent_size: int,
        out_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        *,
        first_omega: float,
        hidden_omega: float,
        modulated_layers: list[int],
        outermost_linear: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.outermost_linear = outermost_linear
        layers: list[nn.Module] = []
        l_in_mod = 0 in modulated_layers
        layers.append(
            SineLayer(
                in_size,
                latent_size if l_in_mod else 0,
                hidden_size,
                is_first=True,
                omega=first_omega,
            )
        )
        for i in range(num_hidden_layers):
            mod = (i + 1) in modulated_layers
            layers.append(
                SineLayer(
                    hidden_size,
                    latent_size if mod else 0,
                    hidden_size,
                    is_first=False,
                    omega=hidden_omega,
                )
            )
        self.net = nn.Sequential(*layers)
        if outermost_linear:
            self.final = nn.Linear(hidden_size + latent_size, out_size, bias=True)
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_size) / hidden_omega
                self.final.weight.uniform_(-bound, bound)
        else:
            self.final = SineLayer(
                hidden_size, 0, out_size, is_first=False, omega=hidden_omega
            )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h, latent = self.net(x)
        if self.outermost_linear:
            return self.final(torch.cat([h, latent], dim=-1))
        out, _ = self.final((h, latent))
        return out
