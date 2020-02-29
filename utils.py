import time
import numpy
import math
import os
import sys
import matplotlib.pyplot as plt
from skimage import io, img_as_float

import const

def isPointinPolygon(point, rangelist):
    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    #print(lnglist, latlist)
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    #print(maxlng, minlng, maxlat, minlat)
    if (point[0] > maxlng or point[0] < minlng or
        point[1] > maxlat or point[1] < minlat):
        return False
    
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            return False
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            #print(point12lng)
            if (point12lng == point[0]):
                return False
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    #print(count)
    if count%2 == 0:
        return False
    else:
        return True

def showFig(key_points):
    plt.figure()
    x = []
    y = []
    for (x_, y_) in key_points:
        #print(x_)
        #print(y_)
        x.append(x_)
        y.append(y_)
    plt.plot(x, y, 'ro')
    plt.show()
    #plt.savefig("res.jpg")

def getGravity(p):
    center_x = 0.0
    center_y = 0.0
    cnt = 0

    for point in p:
        center_x += point[0]
        center_y += point[1]
        cnt += 1

    return [float(center_x) / cnt, float(center_y) / cnt]

def getBlendedImg(M, background, foreground, output_img_path):
    (h, w, c) = background.shape

    for y in range(h):
        for x in range(w):
            # eqn(1)
            background[y, x, 0] = M[y, x] * int(foreground[y][x][0]) + (1 - M[y, x]) * int(background[y][x][0])
            background[y, x, 1] = M[y, x] * int(foreground[y][x][1]) + (1 - M[y, x]) * int(background[y][x][1])
            background[y, x, 2] = M[y, x] * int(foreground[y][x][2]) + (1 - M[y, x]) * int(background[y][x][2])

    #io.imshow(background)
    #io.show()
    io.imsave(output_img_path, background)

def calEuclDis(v1, v2):
    if len(v1) != len(v2):
        print("len(v1) not equal to len(v2), calEulDis error")
        return -1
    
    ans = 0.0
    for i in range(len(v1)):
        ans += math.sqrt((v1[i][0] - v2[i][0]) * (v1[i][0] - v2[i][0]) + (v1[i][1] - v2[i][1]) * (v1[i][1] - v2[i][1]))

    return ans
