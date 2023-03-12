import cv2
import numpy as np

# Load citra
img = cv2.imread('comel.jpg')

# Konversi citra ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Terapkan min filter dengan ukuran kernel 5x5
kernel_size = 5
filtered = cv2.erode(gray, np.ones((kernel_size,kernel_size),np.uint8))

# Tampilkan citra asli dan citra hasil filter
cv2.imshow('Original', gray)
cv2.imshow('Filtered', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()