import mlx.core as mx


def get_window(window, length) -> mx.array:
    """Generate a window of a specified type.

    Args:
        window (str): Type of window. Supported types are 'rectangular', 'hann', 'blackman'.
        length (int): Length of the window.
        **kwargs: Additional arguments specific to the chosen window type.

    Returns:
        mx.array: Window of specified type.
    """
    if window == "rectangular":
        return rect_window(length)
    elif window == "hann":
        return hann_window(length)
    elif window == "blackman":
        return blackman_window(length)
    else:
        raise ValueError(
            "Unsupported window type. Supported types are 'rectangular', 'hann','blackman'."
        )


def hann_window(length: int) -> mx.array:
    """Generate a Hann window.

    Args:
        length (int): Length of the window.

    Returns:
        mx.array: Hann window of specified length.
    """
    return 0.5 - 0.5 * mx.cos(2 * mx.pi * mx.arange(length) / (length - 1))


def rect_window(length: int) -> mx.array:
    """Generate a rectangular (boxcar) window.

    Args:
        length (int): Length of the window.

    Returns:
        mx.array: Rectangular window of specified length.
    """
    return mx.ones(length)


def blackman_window(length: int) -> mx.array:
    """Generate a Blackman window.

    Args:
        length (int): Length of the window.

    Returns:
        mx.array: Blackman window of specified length.
    """
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    n = mx.arange(length)
    return (
        a0
        - a1 * mx.cos(2 * mx.pi * n / (length - 1))
        + a2 * mx.cos(4 * mx.pi * n / (length - 1))
    )
