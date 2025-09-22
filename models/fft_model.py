# Linear + FFT trend/seasonality reconstructor for (B, L, K) time series
# PyTorch ≥ 1.10 recommended

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import torch.nn.init as init


def _time_grid(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Centered time grid t in [-1, 1] with length L (shape: [L]).
    Centering makes the intercept = mean(y), slope = cov(y, t)/var(t).
    """
    t = torch.linspace(-1.0, 1.0, steps=L, device=device, dtype=dtype)
    return t


def _fourier_bases(L: int, n_freqs: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns cosine and sine bases with shape [L, n_freqs] for frequencies 1..n_freqs:
      cos[:, f-1] = cos(2π f t / L_idx), sin[:, f-1] = sin(2π f t / L_idx)
    where t is integer index [0..L-1] (DFT-consistent), independent of the centered grid used for trend.
    """
    t_idx = torch.arange(L, device=device, dtype=dtype)  # [0, 1, ..., L-1]
    freqs = torch.arange(1, n_freqs + 1, device=device, dtype=dtype)  # 1..n_freqs
    # [L, n_freqs]
    angles = 2.0 * math.pi * (t_idx[:, None] * freqs[None, :]) / float(L)
    cos_basis = torch.cos(angles)
    sin_basis = torch.sin(angles)
    return cos_basis, sin_basis


class LinearFFTDecoder(nn.Module):
    """
    Pure decoder that maps learnable coefficients to a reconstruction y_hat in R^{B x L x K}:
      y_hat = (intercept + slope * t_centered) + Σ_f (a_f cos_f + b_f sin_f)
    Coefficients are provided externally (see LinearFFTCoeffs below).
    """
    def __init__(self, L: int, n_freqs: int, device=None, dtype=None):
        super().__init__()
        self.L = L
        self.n_freqs = n_freqs
        # Trend grid (centered)
        t = _time_grid(L, device=device, dtype=dtype)                 # [L]
        self.register_buffer("t_centered", t, persistent=False)
        # Fourier bases (DFT-consistent)
        cos_b, sin_b = _fourier_bases(L, n_freqs, device=device, dtype=dtype)  # [L, n_freqs] each
        self.register_buffer("cos_basis", cos_b, persistent=False)
        self.register_buffer("sin_basis", sin_b, persistent=False)

    def forward(self, coeffs: "LinearFFTCoeffs") -> torch.Tensor:
        trend, seasonal = self.decompose(coeffs)
        return trend + seasonal

    def decompose(self, coeffs: "LinearFFTCoeffs") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (trend, seasonal), each shaped [B, L, K].
        """
        # Trend: intercept + slope * t_centered
        # intercept, slope: [B, K]
        trend = coeffs.intercept.unsqueeze(1) + coeffs.slope.unsqueeze(1) * self.t_centered[None, :, None]  # [B, L, K]

        # Seasonal via Fourier series coefficients:
        # cos/sin coeffs: [B, K, n_freqs], bases: [L, n_freqs]
        # einsum 'lf,bkf->blk' gives [B, L, K]
        cos_term = torch.einsum("lf,bkf->blk", self.cos_basis, coeffs.cos_coef)
        sin_term = torch.einsum("lf,bkf->blk", self.sin_basis, coeffs.sin_coef)
        seasonal = cos_term + sin_term
        return trend, seasonal


class LinearFFTCoeffs(nn.Module):
    """
    Container for learnable coefficients:
      intercept: [B, K]
      slope:     [B, K]
      cos_coef:  [B, K, n_freqs]
      sin_coef:  [B, K, n_freqs]
    """
    def __init__(self, intercept, slope, cos_coef, sin_coef, random=False):
        super().__init__()
        if random:
          self.intercept = nn.Parameter(torch.empty(intercept.shape))   # [B, K]
          self.slope     = nn.Parameter(torch.empty(slope.shape))       # [B, K]
          self.cos_coef  = nn.Parameter(torch.empty(cos_coef.shape))   # [B, K, n_freqs]
          self.sin_coef  = nn.Parameter(torch.empty(sin_coef.shape))    # [B, K, n_freqs]

          with torch.no_grad():
            init.kaiming_uniform_(self.intercept)
            init.kaiming_uniform_(self.slope)
            init.kaiming_uniform_(self.cos_coef)
            init.kaiming_uniform_(self.sin_coef)
        else:
          self.intercept = nn.Parameter(intercept)   # [B, K]
          self.slope     = nn.Parameter(slope)       # [B, K]
          self.cos_coef  = nn.Parameter(cos_coef)    # [B, K, n_freqs]
          self.sin_coef  = nn.Parameter(sin_coef)    # [B, K, n_freqs]


@dataclass
class FitResult:
    decoder: LinearFFTDecoder
    coeffs: LinearFFTCoeffs
    history: list  # training loss per iteration


def _init_from_trend_fft(y: torch.Tensor, n_freqs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize coefficients from closed-form:
      - Trend: intercept = mean_t(y), slope = cov(y, t_centered)/var(t_centered)
      - Seasonality: use rFFT on the detrended series to set cos/sin coefficients.

    Args:
      y: [B, L, K]
    Returns:
      intercept[B,K], slope[B,K], cos[B,K,n_freqs], sin[B,K,n_freqs]
    """
    B, L, K = y.shape
    device, dtype = y.device, y.dtype

    # ---- Trend init (centered time grid => mean(t)=0) ----
    t = _time_grid(L, device=device, dtype=dtype)                # [L]
    denom = (t * t).sum() + 1e-12                                # scalar
    y_mean = y.mean(dim=1)                                       # [B, K]
    # slope = sum_t t*y / sum_t t^2
    slope = (t[None, :, None] * y).sum(dim=1) / denom            # [B, K]
    intercept = y_mean                                           # [B, K], since E[t]=0

    # Detrend for FFT init
    detrended = y - intercept.unsqueeze(1) - slope.unsqueeze(1) * t[None, :, None]  # [B, L, K]

    # ---- FFT init (map DFT to Fourier series) ----
    # rfft over time dimension (L) -> bins: 0..floor(L/2)

    X = torch.fft.rfft(detrended, dim=1)                         # [B, 1+L//2, K], complex
    max_bins = X.shape[1] - 1                                    # exclude DC (bin 0)
    n_use = min(n_freqs, max_bins)
    if n_use < n_freqs:
        # Pad missing higher freqs with zeros to match requested n_freqs
        pad_needed = n_freqs - n_use
    else:
        pad_needed = 0

    # Take bins 1..n_use
    re = X[:, 1:1 + n_use, :].real                               # [B, n_use, K]
    im = X[:, 1:1 + n_use, :].imag


    # Discrete Fourier series mapping:
    # y_t ≈ a0 + Σ_f [α_f cos(2π f t/L) + β_f sin(2π f t/L)]
    # For real signals, α_f = 2/L * Re(X_f), β_f = -2/L * Im(X_f)  (for 1..floor(L/2))
    scale = 2.0 / float(L)
    alpha = scale * re                                            # [B, n_use, K]
    beta  = -scale * im

    # Arrange to [B, K, n_freqs]
    cos_coef = alpha.permute(0, 2, 1).contiguous()               # [B, K, n_use]
    sin_coef = beta.permute(0, 2, 1).contiguous()                # [B, K, n_use]

    if pad_needed > 0:
        pad_shape = (B, K, pad_needed)
        cos_coef = torch.cat([cos_coef, torch.zeros(pad_shape, device=device, dtype=dtype)], dim=-1)
        sin_coef = torch.cat([sin_coef, torch.zeros(pad_shape, device=device, dtype=dtype)], dim=-1)

    return intercept, slope, cos_coef, sin_coef


def fit_linear_fft(
    y: torch.Tensor,
    y_mask: torch.Tensor,
    n_freqs: int = 16,
    n_iters: int = 1000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    verbose: bool = True
) -> FitResult:
    """
    Fit a Linear + FFT model to y by learning coefficients per (batch, feature).

    Args:
      y:          input series, shape [B, L, K]
      n_freqs:    number of Fourier frequencies (1..n_freqs) to use
      n_iters:    gradient steps
      lr:         Adam learning rate
      weight_decay: L2 on parameters (can help regularize)
      verbose:    print progress every ~10%

    Returns:
      FitResult with decoder, learned coeffs, and training loss history.
    """
    assert y.ndim == 3, "y must be shaped [B, L, K]"
    B, L, K = y.shape
    device, dtype = y.device, y.dtype

    # Cap n_freqs to Nyquist limit
    max_freqs = L // 2
    n_freqs = min(n_freqs, max_freqs)

    # Build decoder and init coefficients
    decoder = LinearFFTDecoder(L=L, n_freqs=n_freqs, device=device, dtype=dtype)

    intercept0, slope0, cos0, sin0 = _init_from_trend_fft(y, n_freqs=n_freqs)
    coeffs = LinearFFTCoeffs(intercept0, slope0, cos0, sin0).to(device=device, dtype=dtype)

    # Optimize coefficients to minimize MSE reconstruction error
    opt = torch.optim.Adam(coeffs.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    log_every = max(1, n_iters // 10)
    pbar = trange(n_iters + 1)
    for step in pbar: #range(1, n_iters + 1):
        opt.zero_grad(set_to_none=True)
        y_hat = decoder(coeffs)                                  # [B, L, K]
        loss = F.mse_loss(y_hat * y_mask, y * y_mask)
        loss.backward()
        opt.step()

        history.append(loss.item())
        # if verbose and (step % log_every == 0 or step == 1 or step == n_iters):
        # print(f"[{step:4d}/{n_iters}] MSE: {loss.item():.6f}")
        pbar.set_postfix({
            'step': f"{step}/{n_iters}",
            'MSE': f"{loss.item():.6f}"
        })

    return FitResult(decoder=decoder, coeffs=coeffs, history=history)

def fit_linear_fft_1(
    y: torch.Tensor,
    y_mask: torch.Tensor,
    n_freqs: int = 16,
    n_iters: int = 1000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    verbose: bool = True,
    random: bool = False
) -> FitResult:
    """
    Fit a Linear + FFT model to y by learning coefficients per (batch, feature).

    Args:
      y:          input series, shape [B, L, K]
      n_freqs:    number of Fourier frequencies (1..n_freqs) to use
      n_iters:    gradient steps
      lr:         Adam learning rate
      weight_decay: L2 on parameters (can help regularize)
      verbose:    print progress every ~10%

    Returns:
      FitResult with decoder, learned coeffs, and training loss history.
    """
    assert y.ndim == 3, "y must be shaped [B, L, K]"
    B, L, K = y.shape
    device, dtype = y.device, y.dtype

    # Cap n_freqs to Nyquist limit
    max_freqs = L // 2
    n_freqs = min(n_freqs, max_freqs)

    # Build decoder and init coefficients
    decoder = LinearFFTDecoder(L=L, n_freqs=n_freqs, device=device, dtype=dtype)

    intercept0, slope0, cos0, sin0 = _init_from_trend_fft(y, n_freqs=n_freqs)
    coeffs = LinearFFTCoeffs(intercept0, slope0, cos0, sin0, random=random).to(device=device, dtype=dtype)

    # Optimize coefficients to minimize MSE reconstruction error
    opt = torch.optim.Adam(coeffs.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    log_every = max(1, n_iters // 10)
    # pbar = trange(n_iters + 1)
    for step in range(1, n_iters + 1):
        opt.zero_grad(set_to_none=True)
        y_hat = decoder(coeffs)                                  # [B, L, K]
        loss = F.mse_loss(y_hat * y_mask, y * y_mask)
        # print(f"y: {y.shape}, {y_hat.shape}, loss: {loss}, {loss.shape}")
        loss.backward()
        opt.step()

        history.append(loss.item())
        # if verbose and (step % log_every == 0 or step == 1 or step == n_iters):
        # print(f"[{step:4d}/{n_iters}] MSE: {loss.item():.6f}")
        # pbar.set_postfix({
        #     'step': f"{step}/{n_iters}",
        #     'MSE': f"{loss.item():.6f}"
        # })

    return FitResult(decoder=decoder, coeffs=coeffs, history=history)