# Example usage of the STFT module

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_stft import STFT

# Read audio file using soundfile
x, fs = sf.read(r"your_audio_file.wav")

# Convert to mx array and reshape
x = mx.array(x.reshape(1, -1))

# Define STFT parameters
n_fft = 1024
win_length = 256
hop_length = 128
return_db = True
onesided = True

# Create an instance of the STFT module
stft = STFT(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    return_db=return_db,
    onesided=onesided,
)

# Compute STFT of the audio signal
y = stft(x)

# Convert to NumPy array
y = np.array(y[0])

# Display the spectrogram
plt.imshow(
    y,
    aspect="auto",
    cmap="inferno",
    origin="lower",
    extent=[0, x.shape[-1] / fs, 0, fs / 2],
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.colorbar(label="Magnitude (dB)")
plt.show()
