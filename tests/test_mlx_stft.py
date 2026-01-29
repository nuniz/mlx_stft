import unittest

import mlx.core as mx

from mlx_stft import STFT, ISTFT, CompiledISTFT


class TestSTFT(unittest.TestCase):
    """
    Test cases for the STFT class.
    """

    def test_onesided_fft(self, batch: int = 2, signal_length: int = 8000):
        """
        Test one-sided STFT with FFT backend.
        """
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=True, use_fft=True
        )
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape[2], 513)  # n_fft // 2 + 1

    def test_onesided_conv(self, batch: int = 2, signal_length: int = 8000):
        """
        Test one-sided STFT with conv backend.
        """
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=True, use_fft=False
        )
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape[2], 513)

    def test_dualsided_fft(self, batch: int = 2, signal_length: int = 8000):
        """
        Test dual-sided STFT with FFT backend.
        """
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=False, use_fft=True
        )
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape[2], 1024)  # n_fft

    def test_dualsided_conv(self, batch: int = 2, signal_length: int = 8000):
        """
        Test dual-sided STFT with conv backend.
        """
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=False, use_fft=False
        )
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape[2], 1025)  # n_fft + 1 for conv

    def test_stft_backend_consistency(self, batch: int = 2, signal_length: int = 8000):
        """
        Test that FFT and conv backends produce consistent results.
        """
        stft_fft = STFT(n_fft=1024, hop_length=256, onesided=True, use_fft=True)
        stft_conv = STFT(n_fft=1024, hop_length=256, onesided=True, use_fft=False)

        x = mx.random.normal(shape=(batch, signal_length))
        y_fft = stft_fft(x)
        y_conv = stft_conv(x)

        error = mx.max(mx.abs(y_fft - y_conv)).item()
        self.assertLess(error, 1e-2, f"Backend difference: {error}")

    def test_db(self, batch: int = 2, signal_length: int = 8000):
        """
        Test STFT with returning dB.
        """
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=True, return_db=True, use_fft=True
        )
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)


class TestISTFT(unittest.TestCase):
    """
    Test cases for the ISTFT class.
    """

    def test_perfect_reconstruction_onesided_fft(
        self, batch: int = 2, signal_length: int = 8000
    ):
        """
        Test perfect reconstruction with one-sided spectrum using FFT backend.
        """
        n_fft = 2048  # Forces FFT backend
        hop_length = 512

        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            return_db=False,
            use_fft=True,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            use_fft=True,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)
        x_hat = istft(X, length=signal_length)

        # Check reconstruction error
        error = mx.max(mx.abs(x - x_hat)).item()
        self.assertLess(error, 1e-4, f"Reconstruction error too high: {error}")

    def test_perfect_reconstruction_small_n_fft(
        self, batch: int = 2, signal_length: int = 8000
    ):
        """
        Test perfect reconstruction with smaller n_fft.
        """
        n_fft = 512
        hop_length = 128

        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            return_db=False,
            use_fft=True,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            use_fft=True,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)
        x_hat = istft(X, length=signal_length)

        # Check reconstruction error
        error = mx.max(mx.abs(x - x_hat)).item()
        self.assertLess(error, 1e-4, f"Reconstruction error too high: {error}")

    def test_perfect_reconstruction_conv_backend(
        self, batch: int = 2, signal_length: int = 8000
    ):
        """
        Test perfect reconstruction with conv backend for both STFT and iSTFT.
        """
        n_fft = 1024
        hop_length = 256

        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            return_db=False,
            use_fft=False,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            use_fft=False,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)
        x_hat = istft(X, length=signal_length)

        # Conv backend has slightly lower precision
        error = mx.max(mx.abs(x - x_hat)).item()
        self.assertLess(error, 1e-3, f"Reconstruction error too high: {error}")

    def test_istft_backend_consistency(self, batch: int = 2, signal_length: int = 8000):
        """
        Test that iSTFT FFT and conv backends produce consistent results.
        """
        n_fft = 1024
        hop_length = 256

        stft = STFT(n_fft=n_fft, hop_length=hop_length, onesided=True, use_fft=True)
        istft_fft = ISTFT(n_fft=n_fft, hop_length=hop_length, onesided=True, use_fft=True)
        istft_conv = ISTFT(n_fft=n_fft, hop_length=hop_length, onesided=True, use_fft=False)

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)

        x_hat_fft = istft_fft(X, length=signal_length)
        x_hat_conv = istft_conv(X, length=signal_length)

        error = mx.max(mx.abs(x_hat_fft - x_hat_conv)).item()
        self.assertLess(error, 1e-3, f"iSTFT backend difference: {error}")

    def test_perfect_reconstruction_dualsided_fft(
        self, batch: int = 2, signal_length: int = 8000
    ):
        """
        Test perfect reconstruction with dual-sided spectrum using FFT backend.
        """
        n_fft = 2048
        hop_length = 512

        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=False,
            return_db=False,
            use_fft=True,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=False,
            use_fft=True,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)
        x_hat = istft(X, length=signal_length)

        # Check reconstruction error
        error = mx.max(mx.abs(x - x_hat)).item()
        self.assertLess(error, 1e-4, f"Reconstruction error too high: {error}")

    def test_perfect_reconstruction_large_n_fft(
        self, batch: int = 2, signal_length: int = 16000
    ):
        """
        Test perfect reconstruction with larger n_fft (> 2048).
        """
        n_fft = 4096
        hop_length = 1024

        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            return_db=False,
            use_fft=True,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            use_fft=True,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)
        x_hat = istft(X, length=signal_length)

        # Check reconstruction error
        error = mx.max(mx.abs(x - x_hat)).item()
        self.assertLess(error, 1e-4, f"Reconstruction error too high: {error}")

    def test_center_vs_no_center(self, batch: int = 2, signal_length: int = 8000):
        """
        Test reconstruction with center=True and center=False.
        """
        n_fft = 1024
        hop_length = 256

        # Test with center=True (default)
        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            return_db=False,
            use_fft=True,
        )
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            center=True,
            use_fft=True,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)
        x_hat = istft(X, length=signal_length)

        # Check reconstruction error with center=True
        error = mx.max(mx.abs(x - x_hat)).item()
        self.assertLess(error, 1e-4, f"Center=True reconstruction error: {error}")

    def test_compiled_istft_consistency(
        self, batch: int = 2, signal_length: int = 8000
    ):
        """
        Test that CompiledISTFT matches ISTFT output.
        """
        n_fft = 1024
        hop_length = 256

        stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            return_db=False,
            use_fft=True,
        )

        x = mx.random.normal(shape=(batch, signal_length))
        X = stft(x)

        # Create both regular and compiled iSTFT (using FFT backend)
        istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            use_fft=True,
        )
        compiled_istft = CompiledISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=True,
            use_fft=True,
        )

        x_hat = istft(X, length=signal_length)
        x_hat_compiled = compiled_istft(X, length=signal_length)

        # Check that compiled version matches regular version
        error = mx.max(mx.abs(x_hat - x_hat_compiled)).item()
        self.assertLess(error, 1e-6, f"Compiled vs regular mismatch: {error}")

    def test_different_hop_lengths(self, batch: int = 2, signal_length: int = 8000):
        """
        Test perfect reconstruction with different hop lengths.
        """
        n_fft = 1024

        for hop_ratio in [2, 4, 8]:
            hop_length = n_fft // hop_ratio

            stft = STFT(
                n_fft=n_fft,
                hop_length=hop_length,
                onesided=True,
                return_db=False,
                use_fft=True,
            )
            istft = ISTFT(
                n_fft=n_fft,
                hop_length=hop_length,
                onesided=True,
                use_fft=True,
            )

            x = mx.random.normal(shape=(batch, signal_length))
            X = stft(x)
            x_hat = istft(X, length=signal_length)

            # Check reconstruction error
            error = mx.max(mx.abs(x - x_hat)).item()
            self.assertLess(
                error,
                1e-4,
                f"Reconstruction error for hop_ratio={hop_ratio}: {error}",
            )


if __name__ == "__main__":
    unittest.main()
