import mlx.core as mx
import mlx.nn as nn


class AmpToDB(nn.Module):
    def __init__(self, eps: float = 1e-5, top_db: float = 80.0) -> None:
        """
        Initializes the AmpToDB module.

        Arguments:
            eps {float} -- Small value to avoid numerical instability. (default: 1e-5)
            top_db {float} -- Threshold the output at ``top_db`` below the peak (default: 80.0)
        """
        super().__init__()
        self.eps = eps
        self.top_db = top_db

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of the AmpToDB module.

        Arguments:
            x {mx.array} -- Input tensor.

        Returns:
            mx.array -- Output tensor in dB scale.
        """
        x_db = 20 * mx.log10(x.abs() + self.eps)
        max_vals = x_db.max(-1).values
        return mx.max(x_db, (max_vals - self.top_db).unsqueeze(-1))
