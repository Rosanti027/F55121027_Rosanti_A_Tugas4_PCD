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

# Hitung filter Ideal Lowpass dan Butterworth Lowpass
filter_ideal = np.zeros((rows,cols))
filter_ideal[d <= cutoff] = 1

filter_butterworth = 1 / (1 + (d/cutoff)**(2*2))

# Hitung DFT dan geser frekuensi nol ke tengah
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# Terapkan filter dan hitung DFT balik
dft_ideal_shift = dft_shift * filter_ideal
dft_ideal = np.fft.ifftshift(dft_ideal_shift)
img_ideal = np.fft.ifft2(dft_ideal)
img_ideal = np.abs(img_ideal)

dft_butterworth_shift = dft_shift * filter_butterworth
dft_butterworth = np.fft.ifftshift(dft_butterworth_shift)
img_butterworth = np.fft.ifft2(dft_butterworth)
img_butterworth = np.abs(img_butterworth)

# Tampilkan hasil
plt.figure(figsize=(18,6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_ideal, cmap='gray')
plt.title('Citra hasil Ideal Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_butterworth, cmap='gray')
plt.title('Citra hasil Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()