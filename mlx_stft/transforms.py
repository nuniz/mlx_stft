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
        return_db: bool = False,
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
        self.return_db = return_db
        self.window = window
        self.fourier_conv = nn.Identity()
        self.compression = nn.Identity()
        self.initialize(**kwargs)

    def initialize(self, **kwargs) -> None:
        _window = get_window(window=self.window, length=self.win_length)
        _window = pad_signal(_window, target_length=self.n_fft)
        _fourier_basis = precompute_fourier_basis(
            window_size=self.n_fft,
            n_fft=self.n_fft // 2 if self.onesided else self.n_fft,
        )
        _fourier_basis *= _window
        _fourier_basis = _fourier_basis.reshape(-1, _fourier_basis.shape[-1], 1)

        self.fourier_conv = FrozenConv1dLayer(weight=_fourier_basis, stride=self.hop_length, padding=self.n_fft//2)
        self.compression = nn.Identity() if not self.return_db else AmpToDB(dim=1, **kwargs)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.reshape(*x.shape, -1)
        x = self.fourier_conv(x)
        x = x.reshape(x.shape[0], 2, -1 ,x.shape[-1])
        x = self.compression(x)
        return x
