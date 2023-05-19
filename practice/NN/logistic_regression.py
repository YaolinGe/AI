"""
This script rewrites the logistic regression for practice purposes

Author: Yaolin Ge
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image 
from scipy import ndimage
from user_func import load_dataset

# s1, load dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# s2, check one example
index = 10
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

index