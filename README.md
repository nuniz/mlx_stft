# Short-Time Fourier Transform (STFT)

This module implements the Short-Time Fourier Transform (STFT) in Python using MLX.
It is designed to inherit from the nn.Module class, which allows it to be used either as a standalone module or as part of a larger neural network architecture.

## Features

- Compute STFT of audio signals.
- Customizable window functions (e.g., Hann, Hamming).
- Option to return the result in decibels (dB).

## Installation
Install mlx_stft. 
```bash
git clone https://github.com/nuniz/mlx_stft.git
cd mlx_stft
python setup.py install
```

Then, import the `STFT` class from the module.
```python
from mlx_stft import STFT
```

## STFT Arguments
    n_fft: Number of Fourier transform points.
    win_length: Length of the STFT window.
    hop_length: Number of audio samples between adjacent STFT columns.
    window: Type of window function to apply (default is "hann").
    onesided: Whether to return only the non-redundant part of the spectrum (default is False).
    return_db: Whether to return the result in decibels (default is False).

## Usage
```python

# Create an instance of the STFT module
stft = STFT(n_fft=1024, win_length=256, hop_length=128, return_db=True, onesided=True)

# Compute STFT of an audio signal
audio_stft = stft(audio_signal)
```
