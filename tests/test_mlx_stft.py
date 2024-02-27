import unittest

from mlx_stft import STFT


class TestSTFT(unittest.TestCase):
    """
    Test cases for the STFT class.
    """

    def test_onesided(self, batch: int = 2, signal_length: int = 8000):
        """
        Test one-sided STFT.

        Args:
            batch (int): Number of batches.
            signal_length (int): Length of the signal.
        """

        # Create STFT instance
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=True, return_db=False
        )

        # Apply STFT to signal
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)

    def test_dualsided(self, batch: int = 2, signal_length: int = 8000):
        """
        Test dual-sided STFT.

        Args:
            batch (int): Number of batches.
            signal_length (int): Length of the signal.
        """

        # Create STFT instance
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=False, return_db=False
        )

        # Apply STFT to signal
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)

    def test_db(self, batch: int = 2, signal_length: int = 8000):
        """
        Test STFT with returning dB.

        Args:
            batch (int): Number of batches.
            signal_length (int): Length of the signal.
        """

        # Create STFT instance
        stft = STFT(
            n_fft=1024, win_length=512, hop_length=256, onesided=True, return_db=True
        )

        # Apply STFT to signal
        x = mx.random.normal(shape=(batch, signal_length))
        y = stft(x)
        self.assertIsNotNone(y)


if __name__ == "__main__":
    unittest.main()
