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


def pad_signal(x: mx.array, target_length) -> mx.array:
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
        pass
    elif current_length > target_length:
        raise ValueError(
            "Target length should be greater than the current length of the signal."
        )
    else:
        padding = mx.zeros(*x.shape[:-1], target_length - current_length)
        x = mx.concatenate((x, padding))
    return x


def precompute_fourier_basis(window_size: int, n_fft: int) -> mx.array:
    """
    Precompute the Fourier basis.

    Args:
        window_size (int): Size of the window.
        n_fft (int): Number of Fourier transform points.

    Returns:
        mx.array: Precomputed Fourier basis.
    """
    basis_grid = mx.outer(mx.arange(n_fft + 1), mx.arange(window_size))
    basis_grid = 2 * mx.pi * basis_grid / window_size
    basis = mx.stack((mx.cos(basis_grid), mx.sin(basis_grid)), axis=0)
    return basis


def norm(x, dim: int = -1):
    """
    Compute the L2 norm along a specified dimension.

    Args:
        x (mx.array): Input tensor.
        dim (int, optional): Dimension along which to compute the norm. Defaults to -1.

    Returns:
        mx.array: L2 norm along the specified dimension.
    """
    x = mx.sqrt(mx.sum(x**2), dim)
    return x


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
        x = 20 * mx.log10(norm(x, self.dim) + self.eps)
        max_vals = x.max(-1) - self.top_db
        x = mx.maximum(x, max_vals.reshape(*max_vals.shape, 1))
        return x
