import mlx.core as mx
import mlx.nn as nn


class FrozenConv1dLayer(nn.Module):
    """
    Frozen Convolutional 1D layer.

    Args:
        weight (mx.array): The weight tensor for convolution.
        stride (int): The stride of the convolution operation. Defaults to 1.
        padding (int): The padding to be applied. Defaults to 0.
    """

    def __init__(
        self,
        weight: mx.array,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self._weight = weight

        self.padding = padding
        self.stride = stride

    def _extra_repr(self) -> str:
        """
        Returns representation of the layer.

        Returns:
            str: representation string.
        """
        return (
            f"{self._weight.shape[-1]}, {self._weight.shape[0]}, "
            f"kernel_size={self._weight.shape[1]}, stride={self.stride}, "
            f"padding={self.padding}"
        )

    def __call__(self, x) -> mx.array:
        """
        Forward pass of the convolutional layer.

        Args:
            x (mx.array): Input tensor.

        Returns:
            mx.array: Convolved output.
        """
        x = mx.conv1d(x, self._weight, self.stride, self.padding)
        return x


def _overlap_add_single(
    frames: mx.array, flat_indices: mx.array, valid_mask: mx.array, output_length: int
) -> mx.array:
    """
    Overlap-add for a single sample (no batch dimension).

    Args:
        frames (mx.array): Flattened frames [num_frames * frame_length].
        flat_indices (mx.array): Pre-computed scatter indices.
        valid_mask (mx.array): Mask for valid indices.
        output_length (int): Length of output signal.

    Returns:
        mx.array: Reconstructed signal [output_length].
    """
    masked_frames = frames * valid_mask.astype(frames.dtype)
    output = mx.zeros((output_length,))
    output = output.at[flat_indices].add(masked_frames)
    return output


def overlap_add(frames: mx.array, hop_length: int, output_length: int) -> mx.array:
    """
    Perform overlap-add of frames to reconstruct the signal.

    Args:
        frames (mx.array): Input frames of shape [batch, num_frames, frame_length].
        hop_length (int): Number of samples between frame starts.
        output_length (int): Length of the output signal.

    Returns:
        mx.array: Reconstructed signal of shape [batch, output_length].
    """
    batch_size, num_frames, frame_length = frames.shape

    # Pre-compute indices
    frame_starts = mx.arange(num_frames) * hop_length
    sample_indices = mx.arange(frame_length)
    indices = frame_starts[:, None] + sample_indices[None, :]
    flat_indices = indices.reshape(-1)

    # Clip and mask
    valid_mask = flat_indices < output_length
    flat_indices_clipped = mx.clip(flat_indices, 0, output_length - 1)

    # Flatten frames: [batch, num_frames * frame_length]
    frames_flat = frames.reshape(batch_size, -1)

    # Use vmap to parallelize across batch dimension
    # This avoids creating huge intermediate index arrays
    vmapped_overlap_add = mx.vmap(
        lambda f: _overlap_add_single(f, flat_indices_clipped, valid_mask, output_length)
    )
    output = vmapped_overlap_add(frames_flat)

    return output


def compute_window_norm(
    window: mx.array, hop_length: int, output_length: int, num_frames: int
) -> mx.array:
    """
    Compute sum of squared windows at each output position for COLA normalization.

    Args:
        window (mx.array): Window function of shape [frame_length].
        hop_length (int): Number of samples between frame starts.
        output_length (int): Length of the output signal.
        num_frames (int): Number of frames.

    Returns:
        mx.array: Window normalization factor of shape [output_length].
    """
    frame_length = window.shape[0]
    window_sq = mx.square(window)

    # Accumulate squared window at each position
    frame_starts = mx.arange(num_frames) * hop_length
    sample_indices = mx.arange(frame_length)
    indices = frame_starts[:, None] + sample_indices[None, :]

    # Use tile instead of broadcast_to for better materialization behavior
    flat_values = mx.tile(window_sq, (num_frames,))

    # Flatten for scatter
    flat_indices = indices.reshape(-1)

    # Clip indices
    valid_mask = flat_indices < output_length
    flat_indices = mx.clip(flat_indices, 0, output_length - 1)
    flat_values = flat_values * valid_mask.astype(flat_values.dtype)

    # Scatter-add
    norm = mx.zeros((output_length,))
    norm = norm.at[flat_indices].add(flat_values)

    return norm


def precompute_inverse_fourier_basis(
    window_size: int, n_fft: int, window: mx.array, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """
    Precompute the inverse Fourier basis with synthesis window for conv_transpose path.

    Args:
        window_size (int): Size of the window.
        n_fft (int): Number of Fourier transform points.
        window (mx.array): Synthesis window.
        dtype (mx.Dtype): Data type for the basis. Defaults to float32.

    Returns:
        mx.array: Precomputed inverse Fourier basis of shape [2, freq_bins, window_size].
    """
    freq_bins = n_fft + 1
    time_bins = mx.arange(window_size, dtype=dtype)
    freq_indices = mx.arange(freq_bins, dtype=dtype)

    # Compute angular frequencies: 2π * k * n / N
    basis_grid = mx.outer(freq_indices, time_bins) * (2.0 * mx.pi / window_size)

    # Inverse DFT: x[n] = (1/N) * sum_k X[k] * e^{j*2*pi*k*n/N}
    # Real part: x[n] = (1/N) * sum_k (Re[X[k]]*cos - Im[X[k]]*sin)
    # So we need cos for real and -sin for imaginary
    scale = 1.0 / window_size
    cos_basis = mx.cos(basis_grid) * scale * window
    sin_basis = -mx.sin(basis_grid) * scale * window  # Negative for iDFT

    # Stack: [2, freq_bins, window_size]
    return mx.stack((cos_basis, sin_basis), axis=0)


def pad_signal(x: mx.array, target_length: int) -> mx.array:
    """
    Pad a signal with zeros to match the target length.

    Args:
        x (mx.array): Input signal to be padded.
        target_length (int): Length to which the signal will be padded.

    Returns:
        mx.array: Padded signal.
    """
    current_length = x.shape[-1]
    if current_length == target_length:
        return x
    elif current_length > target_length:
        raise ValueError(
            "Target length should be greater than the current length of the signal."
        )
    pad_amount = target_length - current_length
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_amount)]
    return mx.pad(x, pad_width)


def precompute_fourier_basis(
    window_size: int, n_fft: int, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """
    Precompute the Fourier basis.

    Args:
        window_size (int): Size of the window.
        n_fft (int): Number of Fourier transform points.
        dtype (mx.Dtype): Data type for the basis. Defaults to float32.

    Returns:
        mx.array: Precomputed Fourier basis.
    """
    freq_bins = mx.arange(n_fft + 1, dtype=dtype)
    time_bins = mx.arange(window_size, dtype=dtype)
    # Compute angular frequencies: 2π * k * n / N
    basis_grid = mx.outer(freq_bins, time_bins) * (2.0 * mx.pi / window_size)
    # Stack cos (real) and -sin (imaginary) components
    # DFT: X[k] = sum_n x[n] * e^{-j*2*pi*k*n/N} = sum_n x[n] * (cos - j*sin)
    return mx.stack((mx.cos(basis_grid), -mx.sin(basis_grid)), axis=0)


def norm(x: mx.array, dim: int = -1) -> mx.array:
    """
    Compute the L2 norm along a specified dimension.

    Args:
        x (mx.array): Input tensor.
        dim (int, optional): Dimension along which to compute the norm. Defaults to -1.

    Returns:
        mx.array: L2 norm along the specified dimension.
    """
    return mx.sqrt(mx.sum(mx.square(x), axis=dim))


def frame_signal(
    x: mx.array, frame_length: int, hop_length: int, pad_end: bool = True
) -> mx.array:
    """
    Slice a signal into overlapping frames.

    Args:
        x (mx.array): Input signal of shape [batch, length].
        frame_length (int): Length of each frame.
        hop_length (int): Number of samples between frame starts.
        pad_end (bool): Whether to pad the end to include all samples.

    Returns:
        mx.array: Framed signal of shape [batch, num_frames, frame_length].
    """
    _, signal_length = x.shape

    # Pad signal to ensure we capture all samples
    if pad_end:
        pad_amount = frame_length // 2
        x = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])
        signal_length = x.shape[1]

    # Calculate number of frames
    num_frames = 1 + (signal_length - frame_length) // hop_length

    # Create frame indices using broadcasting
    frame_starts = mx.arange(num_frames) * hop_length
    sample_indices = mx.arange(frame_length)
    # Shape: [num_frames, frame_length]
    indices = frame_starts[:, None] + sample_indices[None, :]

    # Gather frames for each batch element
    # x[batch, indices] -> [batch, num_frames, frame_length]
    frames = mx.take(x, indices, axis=1)
    return frames


class AmpToDB(nn.Module):
    def __init__(self, eps: float = 1e-5, top_db: float = 80.0, dim: int = -1) -> None:
        """
        Module to convert linear magnitude to decibel (dB) scale.

        Args:
            eps (float): Small value to avoid numerical instability. Defaults to 1e-5.
            top_db (float): Threshold the output at 'top_db' below the peak. Defaults to 80.0.
            dim (int): Dimension along which to compute the norm. Defaults to -1.
        """
        super().__init__()
        self.eps = eps
        self.top_db = top_db
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the AmpToDB module.

        Args:
            x (mx.array): Input tensor.

        Returns:
            mx.array: Output tensor in dB scale.
        """
        # Compute magnitude in dB: 20 * log10(|x| + eps)
        x = 20.0 * mx.log10(norm(x, self.dim) + self.eps)
        # Use keepdims to avoid reshape allocation
        max_vals = mx.max(x, axis=-1, keepdims=True) - self.top_db
        return mx.maximum(x, max_vals)
