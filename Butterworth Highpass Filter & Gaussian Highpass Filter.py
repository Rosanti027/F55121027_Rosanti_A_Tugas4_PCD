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

# Tentukan cutoff frequency
cutoff = 30

# Hitung filter Butterworth Highpass dan Gaussian Highpass
filter_butterworth = 1 / (1 + (cutoff/d)**(2*2))

filter_gaussian = 1 - np.exp(-(d**2) / (2*(cutoff**2)))

# Hitung DFT dan geser frekuensi nol ke tengah
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# Terapkan filter dan hitung DFT balik
dft_butterworth_shift = dft_shift * filter_butterworth
dft_butterworth = np.fft.ifftshift(dft_butterworth_shift)
img_butterworth = np.fft.ifft2(dft_butterworth)
img_butterworth = np.abs(img_butterworth)

dft_gaussian_shift = dft_shift * filter_gaussian
dft_gaussian = np.fft.ifftshift(dft_gaussian_shift)
img_gaussian = np.fft.ifft2(dft_gaussian)
img_gaussian = np.abs(img_gaussian)

# Tampilkan hasil
plt.figure(figsize=(18,6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_butterworth, cmap='gray')
plt.title('Citra hasil Butterworth Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_gaussian, cmap='gray')
plt.title('Citra hasil Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()