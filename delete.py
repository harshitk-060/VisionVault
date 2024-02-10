
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def lpq(img: np.ndarray, winSize: int = 3, freqestim: int = 1, mode: str = 'nh') -> np.ndarray:
    """
    Compute LPQ (Local Phase Quantization) descriptor for the given image.

    Parameters:
    - img (np.ndarray): Input image (2D numpy array).
    - winSize (int): Size of the window for computing LPQ. Default is 3.
    - freqestim (int): Frequency estimation method:
        - 1: STFT uniform window (default).
    - mode (str): Output mode:
        - 'nh': Normalized histogram (default).
        - 'h': Histogram.
        - 'im': LPQ code image.

    Returns:
    - LPQdesc (np.ndarray): LPQ descriptor.

    Raises:
    - ValueError: If an unsupported frequency estimation method or mode is provided.
    """
    rho = 0.90
    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)
    convmode = 'valid'

    img = np.float64(img)
    r = (winSize - 1) / 2
    x = np.arange(-r, r + 1)[np.newaxis]

    if freqestim == 1:
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)

    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                         filterResp2.real, filterResp2.imag,
                         filterResp3.real, filterResp3.imag,
                         filterResp4.real, filterResp4.imag])

    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)
        plt.imshow(LPQdesc, cmap='gray')
        plt.title("LPQ Code Image")
        plt.axis('off')
        plt.show()

    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]

        if mode == 'nh':
            LPQdesc = LPQdesc / LPQdesc.sum()
        
        plt.bar(np.arange(len(LPQdesc)), LPQdesc)
        plt.title("LPQ Histogram")
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.show()

    return LPQdesc

# Example usage:
# lpq_result = lpq(image, mode='h')  # For histogram visualization
# lpq_result = lpq(image, mode='im') # For LPQ code image visualization

l=lpq(img)