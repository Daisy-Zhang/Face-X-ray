import matplotlib
matplotlib.use('Agg')

import os
from skimage import io
import numpy
import time
import os
import sys
import math

from get_face_alignment import getFaceAlignment
from nearest_search import nearestSearch
from convex_hull import force, mySort
from utils import getGravity, isPointinPolygon, getBlendedImg
from Gaussian_blur import MyGaussianBlur

class X_ray_generator(object):
    background_img_path = ''
    img_db_path =  ''
    output_img_path = ''
    output_xray_path = ''
    h = w = c = 0
    
    background_img_name = ''

    background = None
    mask = None
    xray = None

    key_points = None

    foreground_img_path = ''
    foreground = None

    convex_hull_boundary = None
    M = None
    final_M = None
    B = None

    def __init__(self, bip, idp, oip, oxp):
        super().__init__()
        self.background_img_path = bip
        self.img_db_path = idp
        self.output_img_path = oip
        self.output_xray_path = oxp

    def getBackgroundImgName(self):
        tmp_list = self.background_img_path.split('/')
        self.background_img_name = tmp_list[len(tmp_list) - 1]

    def read(self):
        self.background = io.imread(self.background_img_path)
        self.mask = io.imread(self.background_img_path)
        self.xray = io.imread(self.background_img_path)
        (self.h, self.w, self.c) = self.background.shape
    
    def getKeyPoints(self):
        self.key_points = getFaceAlignment(self.background)

    def getForegroundImg(self):
        self.foreground_img_path = nearestSearch(self.key_points[0], self.img_db_path, self.h, self.w, self.c, self.background_img_name)
        self.foreground = io.imread(self.foreground_img_path)
        print('Foreground image found: ' + self.foreground_img_path)

    def getSortedConvexHullBoundary(self):
        self.convex_hull_boundary = force(list(self.key_points[0]), len(self.key_points[0]))
        for p in self.convex_hull_boundary:
            p[1] = self.h - p[1]
        
        center = getGravity(self.convex_hull_boundary)
        mySort(self.convex_hull_boundary, center)

    def calM(self):
        self.M = numpy.zeros((self.h, self.w))
        #print(M)
        for y in range(self.h): # row
            for x in range(self.w): # col
                if isPointinPolygon([x, y], self.convex_hull_boundary):
                    self.M[y, x] = 1.0
                    self.mask[y, x: ] = 255
                else:
                    self.M[y, x] = 0.0
                    self.mask[y, x: ] = 0
    
    def calGaussianBlur(self):
        self.final_M = MyGaussianBlur(self.M)
    
    def calB(self):
        self.B = numpy.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                # eqn(2)
                self.B[i, j] = 4 * self.final_M[i, j] * (1 - self.final_M[i, j])
                self.xray[i, j: ] = self.B[i, j] * 255.0
                #print(B[i, j])
    
    def dump(self):
        io.imsave(self.output_xray_path, self.xray)
        getBlendedImg(self.M, self.background, self.foreground, self.output_img_path)

    def run(self):
        start_time = time.time()

        self.getBackgroundImgName()
        self.read()
        self.getKeyPoints()
        self.getForegroundImg()
        self.getSortedConvexHullBoundary()

        self.calM()
        self.calGaussianBlur()
        self.calB()

        self.dump()

        end_time = time.time()
        print('Time cost: ', end_time - start_time)
