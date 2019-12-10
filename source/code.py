import numpy as np
import cv2

eiffel = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\eiffel.jpg')
park = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\park.jpg')
selfie = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\selfie.jpg')

cv2.imshow('image', selfie)
cv2.waitKey(0)
cv2.destroyAllWindows()