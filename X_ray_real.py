import face_alignment
import skimage
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from functools import cmp_to_key
import time
import numpy
import math
import os
import sys

img_path = sys.argv[1]
output_img_path = sys.argv[2]
output_xray_path = sys.argv[3]

h = 0
w = 0
c = 0

background = io.imread(img_path)
(h, w, c) = background.shape
B = numpy.zeros((h, w))
io.imsave(output_img_path, background)
io.imsave(output_xray_path, B)