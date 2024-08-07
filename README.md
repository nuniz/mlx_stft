# Short-Time Fourier Transform (STFT)

This module implements the Short-Time Fourier Transform (STFT) in Python using MLX.

It is designed to inherit from the nn.Module, which allows it to be used either as a standalone module or as part of a larger neural network architecture. 

## Installation
You can install mlx_stft directly from pypi:
```bash
pip install mlx_stft
```

Or you can install it directly from the source code:
```bash
git clone https://github.com/nuniz/mlx_stft.git
cd mlx_stft
python setup.py install
```

## Usage
```python
from mlx_stft import STFT

# Create an instance of the STFT module
stft = STFT(n_fft=1024, win_length=256, hop_length=128, return_db=True, onesided=True)

# Compute STFT of an audio signal
y = stft(x)
```

```
x: mx.array [batch, length]
y: mx.array [batch, n_fft // 2, size of fold] if one_sided else [batch, n_fft, size of fold]
where size of fold = 1 + length // hop_length
``` 

## Arguments
    n_fft: Number of Fourier transform points.
    win_length: Length of the STFT window.
    hop_length: Number of audio samples between adjacent STFT columns.
    window: Type of window function to apply (default is "hann").
    onesided: Whether to return only the non-redundant part of the spectrum (default is False).
    return_db: Whether to return the result in decibels (default is False).

## Example

The STFT is illustrated in the picture below using the [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/) dataset's sp09.wav file.

### One-Sided STFT
![one-sided stft](supplementary_material/one-sided.jpg)

### Dual-Sided STFT
![dual-sided stft](supplementary_material/dual-sided.jpg)

