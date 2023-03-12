import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load citra grayscale
img = cv2.imread('comel.jpg', cv2.IMREAD_GRAYSCALE)

# Hitung dimensi citra dan frekuensi nol
rows, cols = img.shape
crow, ccol = rows//2, cols//2

# Hitung jarak frekuensi dari titik tengah
u, v = np.meshgrid(range(cols), range(rows))
d = np.sqrt((u - ccol)**2 + (v - crow)**2)

# Tentukan parameter sigma dan cutoff
sigma = 30
cutoff = 30

# Hitung filter Gaussian Lowpass dan Ideal Highpass
filter_gaussian = np.exp(-(d**2) / (2 * sigma**2))
filter_ideal = np.zeros((rows,cols))
filter_ideal[d > cutoff] = 1

# Hitung DFT dan geser frekuensi nol ke tengah
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# Terapkan filter dan hitung DFT balik
dft_gaussian_shift = dft_shift * filter_gaussian
dft_gaussian = np.fft.ifftshift(dft_gaussian_shift)
img_gaussian = np.fft.ifft2(dft_gaussian)
img_gaussian = np.abs(img_gaussian)

dft_ideal_shift = dft_shift * filter_ideal
dft_ideal = np.fft.ifftshift(dft_ideal_shift)
img_ideal = np.fft.ifft2(dft_ideal)
img_ideal = np.abs(img_ideal)

# Tampilkan hasil
plt.figure(figsize=(18,6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_gaussian, cmap='gray')
plt.title('Citra hasil Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_ideal, cmap='gray')
plt.title('Citra hasil Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()