import cv2
import numpy as np

# Load citra grayscale
img = cv2.imread('comel.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan filter Unsharp Masking
unsharp_mask = cv2.GaussianBlur(img, (0, 0), 2) - img
unsharp_mask = cv2.addWeighted(img, 1.5, unsharp_mask, -0.5, 0)

# Tampilkan citra asli dan hasil filter
cv2.imshow('Citra asli', img)
cv2.imshow('Hasil Unsharp Masking', unsharp_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()