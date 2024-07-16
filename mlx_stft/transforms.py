from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .utils import AmpToDB, FrozenConv1dLayer, pad_signal, precompute_fourier_basis
from .windows import get_window


class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    Args:
        n_fft (int): Number of Fourier transform points.
        win_length (Optional[int]): Length of the STFT window. Defaults to None.
        hop_length (Optional[int]): Number of samples between adjacent STFT columns. Defaults to None.
        window (str): Type of window function to apply. Defaults to "hann".
        onesided (bool): Whether to return only the non-redundant part of the spectrum. Defaults to False.
        return_db (bool): Whether to return the result in decibels. Defaults to False.
        **kwargs: Additional keyword arguments.

    Attributes:
        n_fft (int): Number of Fourier transform points.
        win_length (int): Length of the STFT window.
        hop_length (int): Number of samples between adjacent STFT columns.
        window (str): Type of window function to apply.
        onesided (bool): Whether to return only the non-redundant part of the spectrum.
        return_db (bool): Whether to return the result in decibels.
        fourier_conv (nn.Module): Convolutional layer for computing the STFT.
        compression (nn.Module): Compression layer, if return_db is True.
    """

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
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length

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
        """
        Initialize the STFT module.

        This function precomputes the Fourier basis and sets up the convolutional layers.
        """
        _window = get_window(window=self.window, length=self.win_length)
        _window = pad_signal(_window, target_length=self.n_fft)

        _fourier_basis = precompute_fourier_basis(
            window_size=self.n_fft,
            n_fft=self.n_fft // 2 if self.onesided else self.n_fft,
        )
        _fourier_basis *= _window
        _fourier_basis = _fourier_basis.reshape(-1, _fourier_basis.shape[-1], 1)
        self.fourier_conv = FrozenConv1dLayer(
            weight=_fourier_basis, stride=self.hop_length, padding=self.n_fft // 2
        )

        self.compression = (
            nn.Identity() if not self.return_db else AmpToDB(dim=1, **kwargs)
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Perform the STFT computation on the input.

        Args:
            x (mx.array): Input signal [batch, length].

        Returns:
            mx.array: STFT of the input signal.
            * if not one_sided: [batch, n_fft, size of fold]
            * else: [batch, n_fft // 2, size of fold]
            where size of fold = 1 + length // hop_length
        """
        x = x.reshape(*x.shape, -1)
        x = self.fourier_conv(x).swapaxes(-1, -2)
        x = x.reshape(x.shape[0], 2, -1, x.shape[-1])
        x = self.compression(x)
        return x
