import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load citra grayscale
img = cv2.imread('comel.jpg', cv2.IMREAD_GRAYSCALE)

# Transformasi Fourier 2D
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Filter Laplacian di domain frekuensi
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift = fshift * mask

# Transformasi Fourier terbalik
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

# Tampilkan citra asli dan hasil filter
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Hasil Filter Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()