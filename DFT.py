import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load citra grayscale
img = cv2.imread('comel.jpg', cv2.IMREAD_GRAYSCALE)

# Hitung DFT
dft = np.fft.fft2(img)

# Geser frekuensi nol ke tengah
dft_shift = np.fft.fftshift(dft)

# Hitung magnitudo spektrum
magnitude_spectrum = 20*np.log(np.abs(dft_shift))

# Tampilkan citra dan spektrum frekuensi
plt.figure(figsize=(12,6))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Citra asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Spektrum frekuensi'), plt.xticks([]), plt.yticks([])
plt.show()