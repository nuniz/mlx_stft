import mlx.core as mx
import mlx.nn as nn


class FrozenConv1dLayer(nn.Module):
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
        return (
            f"{self._weight.shape[-1]}, {self._weight.shape[0]}, "
            f"kernel_size={self._weight.shape[1]}, stride={self.stride}, "
            f"padding={self.padding}"
        )

    def __call__(self, x) -> mx.array:
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
    basis_grid = mx.outer(mx.arange(n_fft // 2 + 1), mx.arange(window_size))
    basis_real = mx.cos(2 * mx.pi * basis_grid / window_size)
    basis_imag = mx.sin(2 * mx.pi * basis_grid / window_size)
    basis = mx.stack((basis_real, basis_imag), axis=0)
    return basis


class AmpToDB(nn.Module):
    def __init__(self, eps: float = 1e-5, top_db: float = 80.0, dim:int =0) -> None:
        """
        Initializes the AmpToDB module.

        Arguments:
            eps {float} -- Small value to avoid numerical instability. (default: 1e-5)
            top_db {float} -- Threshold the output at ``top_db`` below the peak (default: 80.0)
        """
        super().__init__()
        self.eps = eps
        self.top_db = top_db
        self.dim=dim

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the AmpToDB module.

        Arguments:
            x {mx.array} -- Input tensor.

        Returns:
            mx.array -- Output tensor in dB scale.
        """
        x = 20 * mx.log10(norm(x, self.dim) + self.eps)
        max_vals = x.max(-1) - self.top_db
        x = mx.maximum(x, max_vals.reshape(*max_vals.shape, 1))
        return x
