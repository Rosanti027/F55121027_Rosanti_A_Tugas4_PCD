import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load citra grayscale
img = cv2.imread('comel.jpg', cv2.IMREAD_GRAYSCALE)

# Transformasi Fourier 2D
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Buat filter Butterworth Highpass
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
D = 30  # radius
n = 4  # order
butterworth_highpass = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
        butterworth_highpass[i, j] = 1 / (1 + (D / dist)**(2*n))

# Buat mask untuk area tertentu
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# Gabungkan filter dengan mask
selective_filter = butterworth_highpass * mask

# Filter citra di domain frekuensi
fshift_filtered = fshift * selective_filter

# Transformasi Fourier terbalik
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

# Tampilkan citra asli dan hasil filter
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Selective Filtering'), plt.xticks([]), plt.yticks([])
plt.show()