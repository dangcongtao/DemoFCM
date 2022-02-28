from skimage.color import rgb2gray
import numpy as np
import cv2

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
from scipy import ndimage
from fcmeans import FCM


# just to demo hơ FCM Works
print("hello\n")

pic = mpimg.imread('/home/peter/2.jpg')  # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)
plt.imshow(pic);


fuzzy_mean = FCM(n_cluster = 3).fit(pic)

# cluster_pic = fuzzy_mean.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
# plt.imshow(cluster_pic)
plt.imshow(fuzzy_mean)
plt.waitforbuttonpress(0)