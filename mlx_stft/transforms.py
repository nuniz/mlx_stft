from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .utils import (
    AmpToDB,
    FrozenConv1dLayer,
    compute_window_norm,
    frame_signal,
    overlap_add,
    pad_signal,
    precompute_fourier_basis,
    precompute_inverse_fourier_basis,
)
from .windows import get_window


# Threshold for switching to FFT-based computation
_FFT_THRESHOLD = 2048


class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    This implementation provides two computational backends:
    - Conv1d-based: Efficient for smaller n_fft (< 2048)
    - FFT-based: More efficient for larger n_fft (>= 2048) due to O(n log n) complexity

    Args:
        n_fft (int): Number of Fourier transform points.
        win_length (Optional[int]): Length of the STFT window. Defaults to None.
        hop_length (Optional[int]): Number of samples between adjacent STFT columns. Defaults to None.
        window (str): Type of window function to apply. Defaults to "hann".
        onesided (bool): Whether to return only the non-redundant part of the spectrum. Defaults to False.
        return_db (bool): Whether to return the result in decibels. Defaults to False.
        use_fft (Optional[bool]): Force FFT-based computation. If None, auto-selects based on n_fft.
        **kwargs: Additional keyword arguments.

    Attributes:
        n_fft (int): Number of Fourier transform points.
        win_length (int): Length of the STFT window.
        hop_length (int): Number of samples between adjacent STFT columns.
        window (str): Type of window function to apply.
        onesided (bool): Whether to return only the non-redundant part of the spectrum.
        return_db (bool): Whether to return the result in decibels.
        fourier_conv (nn.Module): Convolutional layer for computing the STFT (conv mode).
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
        use_fft: Optional[bool] = None,
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

        # Auto-select computation mode based on n_fft size
        if use_fft is None:
            self._use_fft = n_fft >= _FFT_THRESHOLD
        else:
            self._use_fft = use_fft

        self.fourier_conv = nn.Identity()
        self.compression = nn.Identity()
        self._window_tensor: Optional[mx.array] = None

        self.initialize(**kwargs)

    def initialize(self, **kwargs) -> None:
        """
        Initialize the STFT module.

        This function precomputes the Fourier basis and sets up the convolutional layers.
        """
        _window = get_window(window=self.window, length=self.win_length)
        _window = pad_signal(_window, target_length=self.n_fft)

        if self._use_fft:
            # FFT mode: store window tensor for framing
            self._window_tensor = _window
        else:
            # Conv mode: precompute Fourier basis with window applied
            _fourier_basis = precompute_fourier_basis(
                window_size=self.n_fft,
                n_fft=self.n_fft // 2 if self.onesided else self.n_fft - 1,
            )
            _fourier_basis = _fourier_basis * _window
            _fourier_basis = _fourier_basis.reshape(-1, _fourier_basis.shape[-1], 1)
            self.fourier_conv = FrozenConv1dLayer(
                weight=_fourier_basis, stride=self.hop_length, padding=self.n_fft // 2
            )

        self.compression = (
            nn.Identity() if not self.return_db else AmpToDB(dim=1, **kwargs)
        )

    def _forward_conv(self, x: mx.array) -> mx.array:
        """
        Conv1d-based STFT computation.

        Args:
            x (mx.array): Input signal [batch, length].

        Returns:
            mx.array: STFT output [batch, 2, n_fft or n_fft//2+1, num_frames].
        """
        # Add channel dimension: [batch, length] -> [batch, length, 1]
        # MLX conv1d expects input shape (N, L, C_in)
        x = x[:, :, None]

        # Apply Fourier convolution: [batch, num_frames, 2*freq_bins]
        x = self.fourier_conv(x)

        # Reshape to separate real/imag: [batch, 2, freq_bins, num_frames]
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        freq_bins = x.shape[2] // 2
        # Transpose from [batch, num_frames, 2*freq_bins] to [batch, 2*freq_bins, num_frames]
        x = mx.swapaxes(x, 1, 2)
        x = x.reshape(batch_size, 2, freq_bins, num_frames)

        return x

    def _forward_fft(self, x: mx.array) -> mx.array:
        """
        FFT-based STFT computation. More efficient for large n_fft.

        Args:
            x (mx.array): Input signal [batch, length].

        Returns:
            mx.array: STFT output [batch, 2, n_fft or n_fft//2+1, num_frames].
        """
        # Frame the signal: [batch, num_frames, n_fft]
        frames = frame_signal(x, self.n_fft, self.hop_length)

        # Apply window
        frames = frames * self._window_tensor

        # Compute FFT
        if self.onesided:
            # rfft returns [batch, num_frames, n_fft//2 + 1] complex
            spectrum = mx.fft.rfft(frames, axis=-1)
        else:
            # fft returns [batch, num_frames, n_fft] complex
            spectrum = mx.fft.fft(frames, axis=-1)

        # Stack real and imaginary parts first, then single transpose
        # spectrum shape: [batch, num_frames, freq_bins]
        # Stack to [batch, 2, num_frames, freq_bins]
        stacked = mx.stack([spectrum.real, spectrum.imag], axis=1)
        # Single transpose to [batch, 2, freq_bins, num_frames]
        x = mx.swapaxes(stacked, 2, 3)

        return x

    def __call__(self, x: mx.array) -> mx.array:
        """
        Perform the STFT computation on the input.

        Args:
            x (mx.array): Input signal [batch, length].

        Returns:
            mx.array: STFT of the input signal.
            * if not one_sided: [batch, 2, n_fft, num_frames]
            * else: [batch, 2, n_fft // 2 + 1, num_frames]
            where num_frames = 1 + length // hop_length
            The second dimension contains [real, imaginary] components.
        """
        if self._use_fft:
            x = self._forward_fft(x)
        else:
            x = self._forward_conv(x)

        x = self.compression(x)
        return x


class ISTFT(nn.Module):
    """
    Inverse Short-Time Fourier Transform (iSTFT) module.

    This implementation provides two computational backends:
    - Conv-based: Efficient for smaller n_fft (< 2048) using matrix multiplication
    - FFT-based: More efficient for larger n_fft (>= 2048) using irfft/ifft

    Args:
        n_fft (int): Number of Fourier transform points.
        win_length (Optional[int]): Length of the STFT window. Defaults to None (uses n_fft).
        hop_length (Optional[int]): Number of samples between adjacent STFT columns. Defaults to None (win_length // 4).
        window (str): Type of window function to apply. Defaults to "hann".
        onesided (bool): Whether the input is one-sided spectrum. Defaults to False.
        center (bool): Whether to trim center padding from the output. Defaults to True.
        normalized (bool): Whether to apply COLA normalization. Defaults to True.
        use_fft (Optional[bool]): Force FFT-based computation. If None, auto-selects based on n_fft.
        length (Optional[int]): Target output length. Defaults to None.
    """

    def __init__(
        self,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: str = "hann",
        onesided: bool = False,
        center: bool = True,
        normalized: bool = True,
        use_fft: Optional[bool] = None,
        length: Optional[int] = None,
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
        self.center = center
        self.normalized = normalized
        self.window_type = window
        self.default_length = length

        # Auto-select computation mode based on n_fft size
        if use_fft is None:
            self._use_fft = n_fft >= _FFT_THRESHOLD
        else:
            self._use_fft = use_fft

        self._window_tensor: Optional[mx.array] = None
        self._inverse_basis: Optional[mx.array] = None

        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the iSTFT module.

        This function precomputes the window and inverse Fourier basis.
        """
        _window = get_window(window=self.window_type, length=self.win_length)
        _window = pad_signal(_window, target_length=self.n_fft)
        self._window_tensor = _window

        if not self._use_fft:
            # Conv mode: precompute inverse Fourier basis
            # For onesided: n_fft // 2 + 1 bins (DC to Nyquist)
            # For full: n_fft bins
            freq_bins = self.n_fft // 2 + 1 if self.onesided else self.n_fft
            self._inverse_basis = precompute_inverse_fourier_basis(
                window_size=self.n_fft,
                n_fft=freq_bins - 1,  # -1 because precompute adds 1
                window=_window,
            )

        # Precompute onesided scale factor to avoid recreating on every call
        if self.onesided and not self._use_fft:
            freq_bins = self.n_fft // 2 + 1
            if freq_bins > 2:
                self._onesided_scale = mx.concatenate([
                    mx.array([1.0]),
                    mx.ones((freq_bins - 2,)) * 2.0,
                    mx.array([1.0])
                ])
            else:
                self._onesided_scale = mx.ones((freq_bins,))
        else:
            self._onesided_scale = None

    def _forward_fft(self, x: mx.array, length: Optional[int] = None) -> mx.array:
        """
        FFT-based iSTFT computation.

        Args:
            x (mx.array): Input spectrum [batch, 2, freq_bins, num_frames].
            length (Optional[int]): Target output length.

        Returns:
            mx.array: Reconstructed signal [batch, length].
        """
        num_frames = x.shape[3]

        # Single transpose for both real and imag: [batch, 2, freq_bins, num_frames] -> [batch, 2, num_frames, freq_bins]
        x_transposed = mx.swapaxes(x, 2, 3)

        # Extract real and imaginary parts (now [batch, num_frames, freq_bins])
        real_part = x_transposed[:, 0, :, :]
        imag_part = x_transposed[:, 1, :, :]

        # Create complex spectrum
        spectrum = real_part + 1j * imag_part

        # Inverse FFT
        if self.onesided:
            frames = mx.fft.irfft(spectrum, n=self.n_fft, axis=-1)
        else:
            frames = mx.fft.ifft(spectrum, axis=-1).real

        # Apply synthesis window
        frames = frames * self._window_tensor

        # Calculate output length
        if length is not None:
            output_length = length
        elif self.default_length is not None:
            output_length = self.default_length
        else:
            # Estimate from number of frames
            output_length = (num_frames - 1) * self.hop_length + self.n_fft

        # Add padding for center mode
        if self.center:
            pad_amount = self.n_fft // 2
            padded_length = output_length + 2 * pad_amount
        else:
            padded_length = output_length
            pad_amount = 0

        # Overlap-add
        output = overlap_add(frames, self.hop_length, padded_length)

        # COLA normalization
        if self.normalized:
            window_norm = compute_window_norm(
                self._window_tensor, self.hop_length, padded_length, num_frames
            )
            # Avoid division by zero
            window_norm = mx.maximum(window_norm, 1e-8)
            output = output / window_norm

        # Trim center padding
        if self.center:
            output = output[:, pad_amount : pad_amount + output_length]

        return output

    def _forward_conv(self, x: mx.array, length: Optional[int] = None) -> mx.array:
        """
        Conv-based iSTFT computation using matrix multiplication.

        Args:
            x (mx.array): Input spectrum [batch, 2, freq_bins, num_frames].
            length (Optional[int]): Target output length.

        Returns:
            mx.array: Reconstructed signal [batch, length].
        """
        num_frames = x.shape[3]

        # Single transpose for both real and imag: [batch, 2, freq_bins, num_frames] -> [batch, 2, num_frames, freq_bins]
        x_transposed = mx.swapaxes(x, 2, 3)

        # Extract real and imaginary parts (now [batch, num_frames, freq_bins])
        real_part = x_transposed[:, 0, :, :]
        imag_part = x_transposed[:, 1, :, :]

        # Inverse basis: [2, freq_bins, n_fft]
        # We need to compute: sum_k (real * cos_basis + imag * sin_basis)
        cos_basis = self._inverse_basis[0]  # [freq_bins, n_fft]
        sin_basis = self._inverse_basis[1]  # [freq_bins, n_fft]

        # For onesided, we need to account for conjugate symmetry
        if self.onesided and self._onesided_scale is not None:
            # Use precomputed scale array: [1, 2, 2, ..., 2, 1] for onesided
            real_part = real_part * self._onesided_scale
            imag_part = imag_part * self._onesided_scale

        # Perform inverse transform: real * cos + imag * sin
        # [batch, num_frames, freq_bins] @ [freq_bins, n_fft] -> [batch, num_frames, n_fft]
        frames = mx.matmul(real_part, cos_basis) + mx.matmul(imag_part, sin_basis)

        # Calculate output length
        if length is not None:
            output_length = length
        elif self.default_length is not None:
            output_length = self.default_length
        else:
            output_length = (num_frames - 1) * self.hop_length + self.n_fft

        # Add padding for center mode
        if self.center:
            pad_amount = self.n_fft // 2
            padded_length = output_length + 2 * pad_amount
        else:
            padded_length = output_length
            pad_amount = 0

        # Overlap-add
        output = overlap_add(frames, self.hop_length, padded_length)

        # COLA normalization
        if self.normalized:
            window_norm = compute_window_norm(
                self._window_tensor, self.hop_length, padded_length, num_frames
            )
            window_norm = mx.maximum(window_norm, 1e-8)
            output = output / window_norm

        # Trim center padding
        if self.center:
            output = output[:, pad_amount : pad_amount + output_length]

        return output

    def __call__(self, x: mx.array, length: Optional[int] = None) -> mx.array:
        """
        Perform the iSTFT computation on the input.

        Args:
            x (mx.array): Input spectrum [batch, 2, freq_bins, num_frames].
                         The second dimension contains [real, imaginary] components.
            length (Optional[int]): Target output length. If None, uses default_length
                                   or estimates from input shape.

        Returns:
            mx.array: Reconstructed signal [batch, length].
        """
        if self._use_fft:
            return self._forward_fft(x, length)
        else:
            return self._forward_conv(x, length)


# Compiled version for maximum performance
def _create_compiled_stft(stft_module: STFT):
    """
    Create a compiled (JIT) version of STFT forward pass.

    Args:
        stft_module: An initialized STFT module.

    Returns:
        Compiled forward function.
    """
    if stft_module._use_fft:

        @mx.compile
        def compiled_forward(x: mx.array, window: mx.array) -> mx.array:
            frames = frame_signal(x, stft_module.n_fft, stft_module.hop_length)
            frames = frames * window
            if stft_module.onesided:
                spectrum = mx.fft.rfft(frames, axis=-1)
            else:
                spectrum = mx.fft.fft(frames, axis=-1)
            # Stack first, then single transpose
            stacked = mx.stack([spectrum.real, spectrum.imag], axis=1)
            return mx.swapaxes(stacked, 2, 3)

        return compiled_forward
    else:

        @mx.compile
        def compiled_forward(x: mx.array, weight: mx.array, stride: int, padding: int) -> mx.array:
            # MLX conv1d expects input shape (N, L, C_in)
            x = x[:, :, None]
            x = mx.conv1d(x, weight, stride, padding)
            batch_size = x.shape[0]
            num_frames = x.shape[1]
            freq_bins = x.shape[2] // 2
            # Transpose from [batch, num_frames, 2*freq_bins] to [batch, 2*freq_bins, num_frames]
            x = mx.swapaxes(x, 1, 2)
            return x.reshape(batch_size, 2, freq_bins, num_frames)

        return compiled_forward


class CompiledSTFT(nn.Module):
    """
    JIT-compiled STFT for maximum performance.

    This class wraps STFT with MLX's JIT compilation for fused kernel execution.
    Use this when processing many batches with the same parameters.

    Args:
        Same as STFT.
    """

    def __init__(
        self,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: str = "hann",
        onesided: bool = False,
        return_db: bool = False,
        use_fft: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Create base STFT module
        self._stft = STFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            onesided=onesided,
            return_db=return_db,
            use_fft=use_fft,
            **kwargs,
        )

        # Create compiled forward function
        self._compiled_forward = _create_compiled_stft(self._stft)

        # Store references for compiled function arguments
        self._use_fft = self._stft._use_fft
        if self._use_fft:
            self._window = self._stft._window_tensor
        else:
            self._weight = self._stft.fourier_conv._weight
            self._stride = self._stft.fourier_conv.stride
            self._padding = self._stft.fourier_conv.padding

        self.compression = self._stft.compression

    def __call__(self, x: mx.array) -> mx.array:
        """
        Perform JIT-compiled STFT computation.

        Args:
            x (mx.array): Input signal [batch, length].

        Returns:
            mx.array: STFT of the input signal.
        """
        if self._use_fft:
            x = self._compiled_forward(x, self._window)
        else:
            x = self._compiled_forward(x, self._weight, self._stride, self._padding)

        x = self.compression(x)
        return x


def _create_compiled_istft(istft_module: "ISTFT"):
    """
    Create a compiled (JIT) version of iSTFT forward pass.

    Args:
        istft_module: An initialized ISTFT module.

    Returns:
        Compiled forward function.
    """
    n_fft = istft_module.n_fft
    hop_length = istft_module.hop_length
    onesided = istft_module.onesided
    center = istft_module.center
    normalized = istft_module.normalized

    if istft_module._use_fft:

        @mx.compile
        def compiled_forward(
            x: mx.array,
            window: mx.array,
            output_length: int,
            padded_length: int,
            pad_amount: int,
        ) -> mx.array:
            num_frames = x.shape[3]

            # Single transpose for both real and imag
            x_transposed = mx.swapaxes(x, 2, 3)

            # Extract real and imaginary parts (now [batch, num_frames, freq_bins])
            real_part = x_transposed[:, 0, :, :]
            imag_part = x_transposed[:, 1, :, :]

            # Create complex spectrum
            spectrum = real_part + 1j * imag_part

            # Inverse FFT
            if onesided:
                frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
            else:
                frames = mx.fft.ifft(spectrum, axis=-1).real

            # Apply synthesis window
            frames = frames * window

            # Overlap-add
            output = overlap_add(frames, hop_length, padded_length)

            # COLA normalization
            if normalized:
                window_norm = compute_window_norm(window, hop_length, padded_length, num_frames)
                window_norm = mx.maximum(window_norm, 1e-8)
                output = output / window_norm

            # Trim center padding
            if center:
                output = output[:, pad_amount : pad_amount + output_length]

            return output

        return compiled_forward

    else:

        @mx.compile
        def compiled_forward(
            x: mx.array,
            window: mx.array,
            cos_basis: mx.array,
            sin_basis: mx.array,
            onesided_scale: mx.array,
            output_length: int,
            padded_length: int,
            pad_amount: int,
        ) -> mx.array:
            num_frames = x.shape[3]

            # Single transpose for both real and imag
            x_transposed = mx.swapaxes(x, 2, 3)

            # Extract real and imaginary parts (now [batch, num_frames, freq_bins])
            real_part = x_transposed[:, 0, :, :]
            imag_part = x_transposed[:, 1, :, :]

            # For onesided, use precomputed scale
            if onesided:
                real_part = real_part * onesided_scale
                imag_part = imag_part * onesided_scale

            # Perform inverse transform
            frames = mx.matmul(real_part, cos_basis) + mx.matmul(imag_part, sin_basis)

            # Overlap-add
            output = overlap_add(frames, hop_length, padded_length)

            # COLA normalization
            if normalized:
                window_norm = compute_window_norm(window, hop_length, padded_length, num_frames)
                window_norm = mx.maximum(window_norm, 1e-8)
                output = output / window_norm

            # Trim center padding
            if center:
                output = output[:, pad_amount : pad_amount + output_length]

            return output

        return compiled_forward


class CompiledISTFT(nn.Module):
    """
    JIT-compiled iSTFT for maximum performance.

    This class wraps ISTFT with MLX's JIT compilation for fused kernel execution.
    Use this when processing many batches with the same parameters.

    Args:
        Same as ISTFT.
    """

    def __init__(
        self,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: str = "hann",
        onesided: bool = False,
        center: bool = True,
        normalized: bool = True,
        use_fft: Optional[bool] = None,
        length: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Create base ISTFT module
        self._istft = ISTFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            onesided=onesided,
            center=center,
            normalized=normalized,
            use_fft=use_fft,
            length=length,
        )

        # Create compiled forward function
        self._compiled_forward = _create_compiled_istft(self._istft)

        # Store references for compiled function arguments
        self._use_fft = self._istft._use_fft
        self._window = self._istft._window_tensor
        self.n_fft = self._istft.n_fft
        self.hop_length = self._istft.hop_length
        self.center = self._istft.center
        self.default_length = self._istft.default_length

        if not self._use_fft:
            self._cos_basis = self._istft._inverse_basis[0]
            self._sin_basis = self._istft._inverse_basis[1]
            # For non-onesided cases, create a scale of ones (no-op multiplication)
            if self._istft._onesided_scale is not None:
                self._onesided_scale = self._istft._onesided_scale
            else:
                freq_bins = self._istft.n_fft // 2 + 1 if self._istft.onesided else self._istft.n_fft
                self._onesided_scale = mx.ones((freq_bins,))

    def __call__(self, x: mx.array, length: Optional[int] = None) -> mx.array:
        """
        Perform JIT-compiled iSTFT computation.

        Args:
            x (mx.array): Input spectrum [batch, 2, freq_bins, num_frames].
            length (Optional[int]): Target output length.

        Returns:
            mx.array: Reconstructed signal [batch, length].
        """
        num_frames = x.shape[3]

        # Calculate output length
        if length is not None:
            output_length = length
        elif self.default_length is not None:
            output_length = self.default_length
        else:
            output_length = (num_frames - 1) * self.hop_length + self.n_fft

        # Calculate padded length for center mode
        if self.center:
            pad_amount = self.n_fft // 2
            padded_length = output_length + 2 * pad_amount
        else:
            padded_length = output_length
            pad_amount = 0

        if self._use_fft:
            return self._compiled_forward(
                x, self._window, output_length, padded_length, pad_amount
            )
        else:
            return self._compiled_forward(
                x,
                self._window,
                self._cos_basis,
                self._sin_basis,
                self._onesided_scale,
                output_length,
                padded_length,
                pad_amount,
            )
