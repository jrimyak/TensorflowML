#A convolution is a filter that passes over an image, processes it, and extracts features that show a commonality in the image
#Process: Scan every pixel in the image and then look at its neighboring pixels

import cv2 
import numpy as np 
from scipy import misc
i = misc.ascent()

import matplotlib.pyplot as plt 
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
