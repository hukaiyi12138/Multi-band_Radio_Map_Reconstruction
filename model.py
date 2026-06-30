"""
model.py  -  PR-BTD-DULRTC: Physics-Regularised Block-Term Decomposition
              Deep Unrolled Low-Rank Tensor Completion for Radio Map Estimation.

What this implements
====================

The optimisation problem is:

    min_{X, {S_r, c_r}, E, Z}    sum_r g_r(S_r)  +  lambda * ||E||_1
    s.t.   X = sum_r  S_r  o  c_r                              (BTD)
           P_Omega(X + E + Z) = P_Omega(D~)                     (data)
           ||P_Omega(Z)||_F <= delta                            (noise ball)
           S_r >= 0,  c_r >= 0                                  (physical)

where D~ is the (optionally Friis-pre-conditioned) observed radio map,
S_r are spatial-loss fields, c_r are per-emitter power-spectral densities,
and g_r is a learned regulariser implemented by a physics-aware network V_k
that takes ray-cast inputs (line integral of B from TX) and a closed-form
free-space anchor as additional channels.

ADMM produces closed-form updates for X, S_r, c_r, E, Z and the multipliers
Lambda, Gamma_r, Y; the only learned operator is V_k.

Versus the previous EC-DULRTC-RME
=================================
  - Three SVT blocks on mode-1/2/3 unfoldings are GONE.  Inter-frequency
    low-rankness is now enforced exactly via the BTD constraint
    (rank <= R in the frequency mode).  No SVD anywhere.
  - CBAM/FiLM proximal blocks REPLACED by per-emitter SLFProximalBlock /
    UNetProximalBlock fed with physically meaningful per-emitter fields.
  - All ADMM scalars (mu, rho, theta, lambda, delta) trained via softplus.
  - Two physics scalars (n, alpha) in the free-space anchor are also learned.

Gradient-flow fixes vs. earlier drafts
=======================================
  - BatchNorm -> GroupNorm in V_k.  BN with batch size 1 is degenerate
    and silently zeros out gradients through normalisation.
  - V_k's final conv: small-std init (1e-3) instead of exact zeros.  Still
    behaves like ~identity at training start (residual is tiny) but autograd
    actually propagates gradients to the rest of V_k and to the anchor.
  - Last block's P update is skipped (configurable).  In the very last
    iteration, P feeds nothing downstream and would receive zero gradient
    regardless; dropping it saves compute and avoids confusing diagnostics.

V_k backbone: CNN vs U-Net
===========================
  Two interchangeable backbones are provided for the learned proximal
  operator V_k:

    - SLFProximalBlock : the original 3-layer CNN.  Receptive field is
      only ~7 px (3 layers x 3x3 kernel), which is too small to propagate
      information from sparse observations across a 256x256 grid.  This
      is the main suspected cause of underperformance vs. RadioUNet at
      1% sampling density.

    - UNetProximalBlock : an encoder-decoder with skip connections.
      Default n_levels=3 gives a receptive field on the order of 60-100 px
      at 256x256 resolution, letting sparse measurements propagate across
      a much larger fraction of the grid.  Same 5 input channels, same
      "anchor + residual" output semantics, same per-emitter weight
      sharing -- fully drop-in.

  Select via BTD_RPCA_Block(..., prox_backbone="unet") or "cnn".  Default
  is "unet"; pass "cnn" to recover the original behaviour exactly.

Tensor shapes throughout
========================
   D, Omega, X, E, Z, Lam, Y :   [B, K, H, W]    radio-map tensors
   S, P, Gam                 :   [B, R, H, W]    per-emitter SLFs
   c                         :   [B, R, K]       per-emitter PSDs
   tx_pos                    :   [B, R, 2]       grid-normalised (x, y), in [-1, 1]
   building                  :   [B, 1, H, W]
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _inv_softplus(x: float) -> float:
    """Inverse of softplus, so that softplus(_inv_softplus(x)) == x."""
    assert x > 0, "target value must be positive"
    return float(np.log(np.exp(x) - 1.0))


def _gn(num_channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides num_channels."""
    g = min(groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# ---------------------------------------------------------------------------
# Ray-cast shadowing operator
# ---------------------------------------------------------------------------

class RayCastShadowing(nn.Module):
    """
    For each emitter r and each pixel x, the line integral of the building
    map B along the straight segment from t_r to x:

        T_r(x) = (1/L) sum_{l=0..L-1} B( t_r + (l/(L-1)) * (x - t_r) )

    This is the geometric-optics shadowing field.  Differentiable in B and
    parameter-free.

    Implementation: one F.grid_sample call per emitter, sampling at L points
    along each ray and averaging.  Cost ~ O(B * R * H * W * L).
    """

    def __init__(self, n_samples: int = 32):
        super().__init__()
        self.L = int(n_samples)

    def forward(self, B: torch.Tensor, tx_pos: torch.Tensor) -> torch.Tensor:
        """
        B      : [N, 1, H, W]                  building mask in [0, 1]
        tx_pos : [N, R, 2]   (x, y) in [-1, 1] grid-normalised coords
        returns: [N, R, H, W]
        """
        N, _, H, W = B.shape
        R = tx_pos.shape[1]
        L = self.L
        device = B.device

        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pix_grid = torch.stack([xx, yy], dim=-1)                 # [H, W, 2]

        t = torch.linspace(0.0, 1.0, L, device=device)           # [L]

        tx_e  = tx_pos[:, :, None, None, None, :]
        pix_e = pix_grid[None, None, :, :, None, :]
        t_e   = t[None, None, None, None, :, None]
        sample_coords = tx_e + t_e * (pix_e - tx_e)              # [N,R,H,W,L,2]

        out = torch.empty(N, R, H, W, device=device, dtype=B.dtype)
        for r in range(R):
            coords_r = sample_coords[:, r].reshape(N, H, W * L, 2)
            sampled  = F.grid_sample(B, coords_r,
                                     mode="bilinear",
                                     padding_mode="border",
                                     align_corners=True)
            out[:, r] = sampled.reshape(N, H, W, L).mean(dim=-1)
        return out


# ---------------------------------------------------------------------------
# Free-space anchor (learnable physics)
# ---------------------------------------------------------------------------

class FreeSpaceAnchor(nn.Module):
    """
    Closed-form path-loss + line-integral shadowing template, mapped to (0, 1].

        P^free_r(x) =  exp( -n * log10( d_r(x)/d0 + 1 )  -  alpha * T_r(x) )

    Two learnable physical scalars (n, alpha), both positive via softplus.
    Output is in (0, 1]:  peaks at 1 at the TX (d=0, T=0) and decays smoothly
    with distance and obstructed ray length.  Matches the normalised radio
    map domain [0, 1] used throughout the pipeline, which keeps the BTD
    factors (S, c) non-negative and well-conditioned at initialisation.

    For dB-domain data, take torch.log of the anchor (or set n to ~10 and
    let scale=10/ln(10) appear as a fixed multiplier).  We chose the (0, 1]
    form because every RME dataset we benchmark on (RadioMapSeer, BART-LAB,
    SpectrumNet) is normalised to that range before training.
    """

    def __init__(self,
                 n_init:     float = 1.0,
                 alpha_init: float = 1.0,
                 d0:         float = 0.05):
        super().__init__()
        self.n_p     = nn.Parameter(torch.tensor(_inv_softplus(n_init)))
        self.alpha_p = nn.Parameter(torch.tensor(_inv_softplus(alpha_init)))
        self.d0      = d0

    def forward(self, d_r: torch.Tensor, T_r: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        n     = F.softplus(self.n_p)     + eps
        alpha = F.softplus(self.alpha_p) + eps
        log_d = torch.log10(d_r / self.d0 + 1.0)
        return torch.exp(-n * log_d - alpha * T_r)


# ---------------------------------------------------------------------------
# SLF proximal block, CNN backbone (original)
# ---------------------------------------------------------------------------

class SLFProximalBlock(nn.Module):
    """
    Learned proximal operator V_k for the per-emitter SLF.  Original
    3-layer CNN backbone.

    Input channels (per emitter, concatenated):
        0: current SLF estimate + scaled dual           (S~ = S + Gam/theta)
        1: distance map d_r (log-scaled)
        2: ray-cast shadowing T_r
        3: building mask B (broadcast across emitters)
        4: free-space anchor P^free_r

    Output: a small residual correction to be ADDED to the anchor.

    Weights are shared across emitters in a given iteration k.

    Why GroupNorm not BatchNorm: training uses batch size 1, which makes
    BN's per-sample variance degenerate.  GroupNorm is per-sample, so it
    works at any batch size.

    Final conv init: small-std (1e-3) rather than exact zeros, so the
    block starts close to identity (P_r ~= anchor) but autograd actually
    flows gradients into V_k's weights from the first backward pass.

    Receptive field is only ~7 px (3 layers x 3x3 kernel), which is too
    small to propagate information from sparse observations across a
    256x256 grid.  See UNetProximalBlock for a larger-receptive-field
    alternative.
    """

    def __init__(self,
                 hidden:    int = 48,
                 n_layers:  int = 3,
                 gn_groups: int = 8,
                 init_std:  float = 1e-3):
        super().__init__()
        in_ch = 5
        layers = []
        c_prev = in_ch
        for i in range(n_layers):
            layers += [
                nn.Conv2d(c_prev, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(gn_groups, hidden),
                nn.ReLU(inplace=True),
            ]
            c_prev = hidden
        layers.append(nn.Conv2d(hidden, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

        nn.init.normal_(self.net[-1].weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,
                S_tilde: torch.Tensor,
                d_r:     torch.Tensor,
                T_r:     torch.Tensor,
                B:       torch.Tensor,
                anchor:  torch.Tensor) -> torch.Tensor:
        N, R, H, W = S_tilde.shape
        d_log = torch.log1p(d_r)
        B_rep = B.expand(N, R, H, W)
        x = torch.stack([S_tilde, d_log, T_r, B_rep, anchor], dim=2)
        x = x.reshape(N * R, 5, H, W)
        return self.net(x).reshape(N, R, H, W)


# ---------------------------------------------------------------------------
# SLF proximal block, U-Net backbone
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU, twice. Same spatial size in/out."""

    def __init__(self, c_in: int, c_out: int, gn_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            _gn(c_out, gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            _gn(c_out, gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetProximalBlock(nn.Module):
    """
    Learned proximal operator V_k for the per-emitter SLF, U-Net backbone.

    Same 5 input channels as SLFProximalBlock:
        0: current SLF estimate + scaled dual   (S~ = S + Gam/theta)
        1: distance map d_r (log-scaled)
        2: ray-cast shadowing T_r
        3: building mask B (broadcast across emitters)
        4: free-space anchor P^free_r

    Output: residual to be ADDED to the anchor (same semantics as before).

    Architecture: depth-`n_levels` encoder-decoder with skip connections.
    Default n_levels=3 gives a receptive field of roughly 60-100 px at
    256x256 resolution -- enough to span most building footprints and
    let sparse measurements propagate across a meaningful fraction of
    the grid, which the original 3-layer CNN (receptive field ~7 px)
    could not do.

    Weights are shared across emitters in a given iteration k, exactly
    as in SLFProximalBlock: input is reshaped to (N*R, 5, H, W) so the
    same network processes every emitter independently but with the
    same weights.
    """

    def __init__(self,
                 base_ch:   int = 32,
                 n_levels:  int = 3,
                 gn_groups: int = 8,
                 init_std:  float = 1e-3):
        super().__init__()
        in_ch = 5
        self.n_levels = int(n_levels)

        # ---- encoder ----
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        c_prev = in_ch
        ch = base_ch
        enc_chs = []
        for _ in range(self.n_levels):
            self.enc_blocks.append(_ConvBlock(c_prev, ch, gn_groups))
            enc_chs.append(ch)
            self.downs.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))
            c_prev = ch
            ch *= 2

        # ---- bottleneck ----
        self.bottleneck = _ConvBlock(c_prev, ch, gn_groups)

        # ---- decoder ----
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        c_dec = ch
        for skip_ch in reversed(enc_chs):
            self.ups.append(
                nn.ConvTranspose2d(c_dec, skip_ch, 4, stride=2, padding=1)
            )
            # after upsampling, concat with skip -> 2*skip_ch channels in
            self.dec_blocks.append(_ConvBlock(2 * skip_ch, skip_ch, gn_groups))
            c_dec = skip_ch

        # ---- output head ----
        self.out_conv = nn.Conv2d(c_dec, 1, 3, padding=1)
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self,
                S_tilde: torch.Tensor,
                d_r:     torch.Tensor,
                T_r:     torch.Tensor,
                B:       torch.Tensor,
                anchor:  torch.Tensor) -> torch.Tensor:
        N, R, H, W = S_tilde.shape
        d_log = torch.log1p(d_r)
        B_rep = B.expand(N, R, H, W)
        x = torch.stack([S_tilde, d_log, T_r, B_rep, anchor], dim=2)
        x = x.reshape(N * R, 5, H, W)

        # pad H, W up to a multiple of 2**n_levels so down/up-sampling
        # is exactly invertible in spatial size (avoids off-by-one crops)
        factor = 2 ** self.n_levels
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips = []
        h = x
        for enc, down in zip(self.enc_blocks, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            # guard against off-by-one from odd input sizes after padding
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.out_conv(h)
        if pad_h or pad_w:
            out = out[..., :H, :W]

        return out.reshape(N, R, H, W)


# ---------------------------------------------------------------------------
# One BTD-ADMM unrolling block
# ---------------------------------------------------------------------------

class BTD_RPCA_Block(nn.Module):
    """
    One unrolled ADMM iteration.  Pre-softplus initial values, calibrated
    for [0, 1]-normalised data:
        mu = theta = 0.10   (moderate primal penalties)
        rho        = 0.50   (BTD coupling)
        lambda     = 0.01   (sparsity threshold ~ noise floor)
        delta      = 0.10   (noise ball radius)

    prox_backbone selects the architecture for the learned proximal
    operator V_k:
        "unet" (default) : UNetProximalBlock, larger receptive field,
                            targets the 1%-sparsity performance gap.
        "cnn"             : SLFProximalBlock, the original 3-layer CNN.

    For "unet", prox_hidden is reused as base_ch and prox_layers is reused
    as n_levels, so existing CLI flags (--prox-hidden, --prox-layers) keep
    working without renaming.
    """

    def __init__(self,
                 R:             int,
                 prox_hidden:   int   = 48,
                 prox_layers:   int   = 3,
                 gn_groups:     int   = 8,
                 prox_init_std: float = 1e-3,
                 use_prox:      bool  = True,
                 prox_backbone: str   = "unet"):
        super().__init__()
        self.R = int(R)
        self.use_prox = bool(use_prox)
        self.prox_backbone = str(prox_backbone).lower()

        if use_prox:
            if self.prox_backbone == "unet":
                self.prox = UNetProximalBlock(
                    base_ch=prox_hidden,
                    n_levels=prox_layers,
                    gn_groups=gn_groups,
                    init_std=prox_init_std,
                )
            elif self.prox_backbone == "cnn":
                self.prox = SLFProximalBlock(
                    hidden=prox_hidden,
                    n_layers=prox_layers,
                    gn_groups=gn_groups,
                    init_std=prox_init_std,
                )
            else:
                raise ValueError(
                    f"Unknown prox_backbone: {prox_backbone!r}; "
                    f"expected 'unet' or 'cnn'."
                )
        else:
            self.prox = None

        self.mu_p     = nn.Parameter(torch.tensor(_inv_softplus(0.10)))
        self.rho_p    = nn.Parameter(torch.tensor(_inv_softplus(0.50)))
        self.theta_p  = nn.Parameter(torch.tensor(_inv_softplus(0.10)))
        self.lambda_p = nn.Parameter(torch.tensor(_inv_softplus(0.01)))
        self.delta_p  = nn.Parameter(torch.tensor(_inv_softplus(0.10)))

    def _get_params(self):
        eps = 1e-6
        return (
            F.softplus(self.mu_p)     + eps,
            F.softplus(self.rho_p)    + eps,
            F.softplus(self.theta_p)  + eps,
            F.softplus(self.lambda_p) + eps,
            F.softplus(self.delta_p)  + eps,
        )

    @staticmethod
    def _btd_assemble(S: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """X = sum_r S_r o c_r
           S: [N, R, H, W],  c: [N, R, K]   ->   [N, K, H, W]                    """
        return torch.einsum("nrhw,nrk->nkhw", S, c)

    def forward(self,
                X, S, c, E, Z, P,
                Lam, Gam, Y,
                D_proj, Omega,
                d_r, T_r, B, anchor):
        """
        Gauss-Seidel order:  S -> c -> X -> E -> Z -> P -> multipliers.

        Why S, c BEFORE X?  Because X is the consensus variable for the BTD
        constraint X = sum_r S_r o c_r.  Updating S, c first lets the same-
        iteration X "see" the new BTD factors, which means V_k -> P -> S, c
        actually flows through to X within the same iteration.  With the
        opposite order (X first), V_k from the last two unrolling blocks
        receives zero gradient because S, c never feed back into X within
        the same block, and there are no subsequent blocks to carry them.
        """
        mu, rho, theta, lamb, delta = self._get_params()
        eps = 1e-8

        # -------- S_r update (uses OLD X, c, P, Gam) ----------------------
        S_new = torch.empty_like(S)
        for r in range(self.R):
            mask_other    = torch.ones(self.R, device=S.device, dtype=S.dtype)
            mask_other[r] = 0.0
            S_other       = S * mask_other.view(1, self.R, 1, 1)
            X_other_btd   = self._btd_assemble(S_other, c)
            R_r           = X + Y / rho - X_other_btd
            c_r           = c[:, r, :]

            inner = torch.einsum("nkhw,nk->nhw", R_r, c_r)
            c_sq  = (c_r ** 2).sum(dim=-1).view(-1, 1, 1)

            num = rho * inner + theta * (P[:, r] - Gam[:, r] / theta)
            den = rho * c_sq + theta
            S_new[:, r] = num / (den + eps)
        S = S_new

        # -------- c_r update (uses NEW S, OLD X) --------------------------
        c_new = torch.empty_like(c)
        for r in range(self.R):
            mask_other    = torch.ones(self.R, device=S.device, dtype=S.dtype)
            mask_other[r] = 0.0
            S_other       = S * mask_other.view(1, self.R, 1, 1)
            X_other_btd   = self._btd_assemble(S_other, c)
            R_r           = X + Y / rho - X_other_btd
            S_r           = S[:, r]

            inner = torch.einsum("nkhw,nhw->nk", R_r, S_r)
            S_sq  = (S_r ** 2).sum(dim=(-1, -2)).unsqueeze(-1)
            c_new[:, r] = inner / (S_sq + eps)
        c = torch.clamp(c_new, min=0.0)

        # -------- X update (uses NEW S, NEW c) ----------------------------
        X_btd  = self._btd_assemble(S, c)
        Psi_X  = D_proj - E - Z + Lam / mu
        Psi_BT = X_btd - Y / rho
        X = (mu * Psi_X + rho * Psi_BT) / (mu + rho)

        # -------- E update (uses NEW X) -----------------------------------
        Psi_E = D_proj - X - Z + Lam / mu
        E     = torch.sign(Psi_E) * F.relu(torch.abs(Psi_E) - lamb / mu)

        # -------- Z update (noise ball) -----------------------------------
        Psi_Z = D_proj - X - E + Lam / mu
        Z_Om  = Psi_Z * Omega
        norm  = Z_Om.reshape(Z_Om.shape[0], -1).norm(dim=-1) + eps
        scale = torch.clamp(delta / norm, max=1.0).view(-1, 1, 1, 1)
        Z     = Psi_Z * (1.0 - Omega) + Z_Om * scale

        # -------- P update (learned proximal V_k) -------------------------
        if self.use_prox:
            S_tilde  = S + Gam / theta
            residual = self.prox(S_tilde, d_r, T_r, B, anchor)
            P        = anchor + residual

        # -------- multipliers ---------------------------------------------
        X_btd_new = self._btd_assemble(S, c)
        Lam = Lam + mu    * (D_proj - X - E - Z)
        Gam = Gam + theta * (S - P)
        Y   = Y   + rho   * (X - X_btd_new)

        return X, S, c, E, Z, P, Lam, Gam, Y


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class DUSPF_RME(nn.Module):
    """
    Stack of N_iter BTD-ADMM unrolling blocks.

    forward signature:
        D_hat, X, E, S, c = model(D_obs, Omega, B, tx_pos)

    Inputs:
        D_obs : [N, K, H, W]  P_Omega(D)   - zero off-Omega
        Omega : [N, K, H, W]  binary mask
        B     : [N, 1, H, W]  building map in [0, 1]
        tx_pos: [N, R, 2]     (x, y) in [-1, 1] grid-normalised coords

    Returns:
        D_hat    : [N, K, H, W]  reconstructed map  ( = X + E )
        X        : [N, K, H, W]  BTD physical component
        E        : [N, K, H, W]  sparse residual
        S        : [N, R, H, W]  per-emitter SLFs
        c        : [N, R, K]     per-emitter PSDs

    prox_backbone:
        "unet" (default) -- larger receptive field, recommended for
                             sparse-measurement regimes (e.g. 1% mask/fiber).
        "cnn"             -- original 3-layer CNN, smaller/faster.
    """

    def __init__(self,
                 R:               int,
                 K:               int,
                 N_iter:          int   = 10,
                 prox_hidden:     int   = 48,
                 prox_layers:     int   = 3,
                 n_ray_samples:   int   = 32,
                 gn_groups:       int   = 8,
                 prox_init_std:   float = 1e-3,
                 skip_last_prox:  bool  = True,
                 prox_backbone:   str   = "unet"):
        super().__init__()
        self.R = int(R)
        self.K = int(K)
        self.N_iter = int(N_iter)
        self.prox_backbone = str(prox_backbone).lower()

        self.raycast = RayCastShadowing(n_samples=n_ray_samples)
        self.anchor  = FreeSpaceAnchor()
        self.blocks  = nn.ModuleList([
            BTD_RPCA_Block(
                R=R,
                prox_hidden=prox_hidden,
                prox_layers=prox_layers,
                gn_groups=gn_groups,
                prox_init_std=prox_init_std,
                use_prox=(k < N_iter - 1) or (not skip_last_prox),
                prox_backbone=prox_backbone,
            )
            for k in range(N_iter)
        ])

    @staticmethod
    def _distance_maps(tx_pos: torch.Tensor, H: int, W: int) -> torch.Tensor:
        device = tx_pos.device
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return torch.norm(grid[None, None] - tx_pos[:, :, None, None], dim=-1)

    @staticmethod
    def _init_psd(S: torch.Tensor,
                D_obs: torch.Tensor,
                Omega: torch.Tensor) -> torch.Tensor:
        """
        Density-adaptive Tikhonov-regularized least-squares initialization for
        the PSD coefficients c_{r,k}.

        For each frequency band k, we solve the R-dimensional linear system:
            (A_k + eps * I) @ c_k = b_k,
            A_k[i, j] = sum_{Omega_k} S_i * S_j
            b_k[i]    = sum_{Omega_k} S_i * D_k

        The Tikhonov parameter eps adapts to the per-band observation density:
            - sparse  (~1%):  eps small  (~1e-4) -> behaves like joint LS,
                            exploiting inter-emitter coupling for identifiability.
            - dense  (~15%):  eps large  (~1e-2) -> approaches independent LS,
                            better conditioned when SLFs are nearly collinear.

        The interpolation is linear in the per-band sparsity ratio
        rho_k = |Omega_k| / (H*W).

        Args:
            S:     [N, R, H, W]   per-emitter SLF
            D_obs: [N, K, H, W]   sparse observation
            Omega: [N, K, H, W]   binary observation mask

        Returns:
            c:     [N, R, K]      non-negative PSD coefficients
        """
        N, R, H, W = S.shape
        _, K, _, _ = D_obs.shape

        S_flat = S.reshape(N, R, -1)            # [N, R, HW]
        D_flat = D_obs.reshape(N, K, -1)        # [N, K, HW]
        O_flat = Omega.reshape(N, K, -1)        # [N, K, HW]

        # ---- adaptive Tikhonov regularization ----
        eps_min = 1e-4
        eps_max = 1e-2
        rho_min = 0.01
        rho_max = 0.15

        HW = float(H * W)
        rho_per_band = O_flat.sum(dim=-1) / HW  # [N, K]
        alpha = ((rho_per_band - rho_min) / (rho_max - rho_min)).clamp(0.0, 1.0)
        eps_reg = eps_min + (eps_max - eps_min) * alpha  # [N, K]

        eye_R = torch.eye(R, device=S.device, dtype=S.dtype).unsqueeze(0)  # [1, R, R]

        c_per_band = []
        for k in range(K):
            mask_k = O_flat[:, k, :].unsqueeze(1)    # [N, 1, HW]
            Sm = S_flat * mask_k                      # [N, R, HW]
            Dm = D_flat[:, k, :] * O_flat[:, k, :]    # [N, HW]

            # A_k = Sm @ Sm.T,  shape [N, R, R]
            A = torch.bmm(Sm, Sm.transpose(1, 2))
            # b_k = Sm @ Dm,    shape [N, R]
            b = torch.bmm(Sm, Dm.unsqueeze(-1)).squeeze(-1)

            # Per-sample adaptive Tikhonov regularization (this band)
            eps_k = eps_reg[:, k].view(N, 1, 1)        # [N, 1, 1]
            A = A + eps_k * eye_R                       # broadcast to [N, R, R]

            # Solve A c = b  ->  c shape [N, R]
            try:
                c_k = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                # Fallback: pseudoinverse when A is too ill-conditioned
                c_k = torch.bmm(torch.linalg.pinv(A), b.unsqueeze(-1)).squeeze(-1)

            c_per_band.append(c_k)

        c = torch.stack(c_per_band, dim=-1)           # [N, R, K]
        c = torch.clamp(c, min=0.0)

        return c

    def forward(self,
                D_obs:  torch.Tensor,
                Omega:  torch.Tensor,
                B:      torch.Tensor,
                tx_pos: torch.Tensor):
        N, K, H, W = D_obs.shape
        assert K == self.K, f"K mismatch: model={self.K}, data={K}"
        device = D_obs.device

        d_r    = self._distance_maps(tx_pos, H, W)
        T_r    = self.raycast(B, tx_pos)
        anchor = self.anchor(d_r, T_r)

        D_proj = D_obs * Omega
        z_K    = torch.zeros(N, K, H, W, device=device)
        z_R    = torch.zeros(N, self.R, H, W, device=device)

        X      = D_proj.clone()
        E      = z_K.clone()
        Z      = z_K.clone()
        Lam    = z_K.clone()
        Y      = z_K.clone()

        S      = anchor.clone()
        c      = self._init_psd(S, D_proj, Omega)
        P      = anchor.clone()
        Gam    = z_R.clone()

        for blk in self.blocks:
            X, S, c, E, Z, P, Lam, Gam, Y = blk(
                X, S, c, E, Z, P, Lam, Gam, Y,
                D_proj, Omega,
                d_r, T_r, B, anchor,
            )

        D_hat = X + E
        return D_hat, X, E, S, c


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print("PR-BTD-DULRTC smoke test")
    print("=" * 60)

    for backbone in ["unet", "cnn"]:
        print(f"\n{'#' * 60}")
        print(f"# prox_backbone = {backbone!r}")
        print(f"{'#' * 60}")

        for tag, (N, K, H, W, R) in [("BART-Lab-like", (2, 3, 64, 64, 3)),
                                      ("RadioMapSeer-like", (2, 1, 64, 64, 1))]:
            print(f"\n[{tag}]  N={N} K={K} H={H} W={W} R={R}")
            model = DUSPF_RME(R=R, K=K, N_iter=4, prox_backbone=backbone)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  model parameters: {n_params:,}")

            D   = torch.randn(N, K, H, W).abs().clamp(0, 1)
            Om  = (torch.rand(N, K, H, W) < 0.1).float()
            Bld = torch.rand(N, 1, H, W)
            tx  = (torch.rand(N, R, 2) * 2 - 1) * 0.8

            D_hat, X, E, S, c = model(D * Om, Om, Bld, tx)
            print(f"  forward   ok.  D_hat {tuple(D_hat.shape)}  "
                  f"S {tuple(S.shape)}  c {tuple(c.shape)}")

            loss = F.l1_loss(D_hat, D)
            loss.backward()

            n_total = sum(1 for p in model.parameters() if p.requires_grad)
            n_grad  = sum(1 for p in model.parameters()
                          if p.grad is not None and p.grad.abs().max() > 0)
            print(f"  backward  ok.  loss={loss.item():.4f}  "
                  f"params w/ nonzero grad: {n_grad}/{n_total}")

            n_grad_val     = model.anchor.n_p.grad.item()
            alpha_grad_val = model.anchor.alpha_p.grad.item()
            print(f"  anchor.n     grad = {n_grad_val:+.3e}")
            print(f"  anchor.alpha grad = {alpha_grad_val:+.3e}")
            assert abs(n_grad_val) > 0,     "anchor.n received no gradient"
            assert abs(alpha_grad_val) > 0, "anchor.alpha received no gradient"

            if model.blocks[0].use_prox:
                final_w = model.blocks[0].prox.out_conv.weight.grad \
                    if backbone == "unet" \
                    else model.blocks[0].prox.net[-1].weight.grad
                print(f"  V_k[0] final conv grad max = "
                      f"{final_w.abs().max().item():.3e}")
                assert final_w.abs().max().item() > 0, \
                    "V_k final conv received no gradient"

    print("\nAll checks passed.")