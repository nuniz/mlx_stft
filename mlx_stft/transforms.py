import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from utils import AmpToDB
from windows import get_window


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_length: Optional[int],
        hop_length: Optional[int],
        window: str = "hann",
        return_log: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.n_fft = 2 ** math.ceil(math.log2(n_fft))
        assert (
            n_fft >= win_length
        ), f"n_fft ({n_fft}) must be greater than or equal to win_length ({win_length})"
        assert (
            win_length >= hop_length
        ), f"win_length ({win_length}) must be greater than or equal to hop_length ({hop_length})"

        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = n_fft // 4 if hop_length is None else hop_length

        self.pad = int(self.nfft // 2 + 1)
        self._window = get_window(window, length=self.win_length)
        self._window = mx.pad()

        self.compression = nn.Identity() if return_log else AmpToDB(**kwargs)

        _fourier_basis = mx.fft.rfft(mx.eye(self.n_fft))
        self._fourier_conv = nn.Conv1d()

    def __call__(self, x: mx.array) -> mx.array:
        x_length = x.shape[-1]
        num_frames = int(math.ceil((x_length - self.win_length) / self.hop_length)) + 1
        padding = (num_frames - 1) * self.hop_length + self.win_length - x_length
        x = mx.pad()
        x = self._fourier_conv(x)
        x = self.compression(x)
        return x
