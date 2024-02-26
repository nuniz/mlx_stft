import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .utils import AmpToDB, FrozenConv1dLayer, pad_signal, precompute_fourier_basis
from .windows import get_window


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: str = "hann",
        onesided: bool = False,
        return_logscale: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = n_fft // 4 if hop_length is None else hop_length

        assert (
            self.n_fft >= self.win_length
        ), f"n_fft ({self.n_fft}) must be greater than or equal to win_length ({self.win_length})"

        assert (
            self.win_length >= self.hop_length
        ), f"win_length ({self.win_length}) must be greater than or equal to hop_length ({self.hop_length})"

        self.onesided = onesided
        self.return_logscale = return_logscale
        self.window = window
        self.stft_sequential = nn.Identity()
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        _fourier_basis = precompute_fourier_basis(
            window_size=self.n_fft,
            n_fft=self.n_fft // 2 if self.onesided else self.n_fft,
        )
        _window = get_window(window=self.window, length=self.win_length)
        _window = pad_signal(_window, target_length=self.n_fft)

        fourier_conv = nn.Conv1d(
            in_channels=1,
            out_channels=_fourier_basis.shape[0],
            kernel_size=self.window_size,
            stride=self.hop_size,
            bias=False,
        )
        fourier_conv.weight = _fourier_basis
        fourier_conv.freeze()

        compression = nn.Identity() if self.return_logscale else AmpToDB(**kwargs)

        self.stft_sequential = nn.Sequential(fourier_conv, compression)

    def __call__(self, x: mx.array) -> mx.array:
        pad_input = self.n_fft // 2

        x = mx.pad(x, pad_with)
        x = self._fourier_conv(x)
        x = self.compression(x)
        return x
