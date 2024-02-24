import mlx.core as mx

def amp_to_db(
    x: mx.array, eps:float=1e-6, top_db:float=40
) -> mx.array:
    """
    Convert the input tensor from amplitude to decibel scale.

    Arguments:
        x {[mx.array]} -- [Input tensor.]

    Keyword Arguments:
        eps {[float]} -- [Small value to avoid numerical instability.]
                          (default: {1e-6})
        top_db {[float]} -- [threshold the output at ``top_db`` below the peak]
            `             (default: {40})

    Returns:
        [mx.array] -- [Output tensor in decibel scale.]
    """
    x_db = 20 * mx.log10(x.abs() + eps)
    return mx.max(x_db, (x_db.max(-1).values - top_db).unsqueeze(-1))