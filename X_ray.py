import face_alignment
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from functools import cmp_to_key
import time
import numpy
import math
import os
import sys

# 凸包算法
def isin(pi,pj,pk,p0):
    x1 = float(p0[0])
    x2 = float(pj[0])
    x3 = float(pi[0])
    x4 = float(pk[0])
    y1 = float(p0[1])
    y2 = float(pj[1])
    y3 = float(pi[1])
    y4 = float(pk[1])

    k_j0=0
    b_j0=0
    k_k0=0
    b_k0=0
    k_jk=0
    b_jk=0
    perpendicular1=False
    perpendicular2 = False
    perpendicular3 = False
    #pj,p0组成的直线，看pi,pk是否位于直线同一侧

    if x2 - x1 == 0:
    #pj,p0组成直线垂直于X轴时
        t1=(x3-x2)*(x4-x2)
        perpendicular1=True
    else:
        k_j0 = (y2 - y1) / (x2 - x1)
        b_j0 = y1 - k_j0 * x1
        t1 = (k_j0 * x3 - y3 + b_j0) * (k_j0 * x4 - y4 + b_j0)

    #pk,p0组成的直线，看pi,pj是否位于直线同一侧

    if x4 - x1 == 0:
    #pk,p0组成直线垂直于X轴时
        t2=(x3-x1)*(x2-x1)
        perpendicular2=True
    else:
        k_k0 = (y4 - y1) / (x4 - x1)
        b_k0 = y1 - k_k0 * x1
        t2 = (k_k0 * x3 - y3 + b_k0) * (k_k0 * x2 - y2 + b_k0)

    # pj,pk组成的直线，看pi,p0是否位于直线同一侧

    if x4 - x2 == 0:
    # pj,pk组成直线垂直于X轴时
        t3=(x3-x2)*(x1-x2)
        perpendicular3 = True
    else:
        k_jk = (y4 - y2) / (x4 - x2)
        b_jk = y2 - k_jk * x2
        t3 = (k_jk * x3 - y3 + b_jk) * (k_jk * x1 - y1 + b_jk)
    #如果pk，p0,pj，三点位于同一直线时，不能将点删除
    if (k_j0 * x4 - y4 + b_j0)==0 and (k_k0 * x2 - y2 + b_k0)==0 and  (k_jk * x1 - y1 + b_jk)==0 :
          t1=-1
    #如果pk，p0,pj，三点位于同一直线时且垂直于X轴，不能将点删除
    if perpendicular1 and perpendicular2 and perpendicular3:
          t1=-1

    return t1,t2,t3

def force(lis, n):
    #集合S中点个数为3时，集合本身即为凸包集
    if n==3:
        return  lis
    else:
        #集合按纵坐标排序，找出y最小的点p0
        lis.sort(key = lambda x: x[1])
        p0=lis[0]
        #除去p0的其余点集合lis_brute
        lis_brute=lis[1:]
        #temp是用来存放集合需要删除的点在lis_brute内的索引，四个点中如果有一个点在其余三个点组成的三角形内部，则该点一定不是凸包上的点
        temp=[]
        #三重循环找到所有这样在三角形内的点
        for i in range(len(lis_brute)-2):
            pi=lis_brute[i]
            #如果索引i已经在temp中，即pi一定不是凸包上的点，跳过这次循环
            if i in temp:
                continue
            for j in range(i+1,len(lis_brute) - 1):
                pj=lis_brute[j]
                #如果索引j已经在temp中，即pj一定不是凸包上的点，跳过这次循环
                if j in temp:
                    continue
                for k in range(j + 1, len(lis_brute)):
                    pk=lis_brute[k]

                    #如果索引k已经在temp中，即pk一定不是凸包上的点，跳过这次循环
                    if k in temp:
                        continue
                    #判断pi是否在pj,pk,p0构成的三角形内
                    (it1,it2,it3)=isin(pi,pj,pk,p0)
                    if it1>=0 and it2>=0 and it3>=0:
                        if i not in temp:
                           temp.append(i)  
                    # 判断pj是否在pi,pk,p0构成的三角形内
                    (jt1,jt2,jt3)=isin(pj,pi,pk,p0)
                    if jt1>=0 and jt2>=0 and jt3>=0:

                        if j not in temp:
                           temp.append(j)

                    # 判断pk是否在pj,pi,p0构成的三角形内
                    (kt1, kt2, kt3)=isin(pk, pi, pj, p0)
                    if kt1 >= 0 and kt2 >= 0 and kt3 >= 0:

                        if k not in temp:
                            temp.append(k)
       #listlast是最终选出的凸包集合
        lislast=[]
        for coor in lis_brute:
            loc = [i for i, x in enumerate(lis_brute) if all(x == coor)]
            for x in loc:
                ploc = x
            if ploc not in temp:
                lislast.append(coor)
        #将p0加入凸包集合
        lislast.append(p0)
        res = []
        for item in lislast:
            res.append([item[0], item[1]])
        return res

def isPointinPolygon(point, rangelist):
    # 判断是否在外包矩形内，如果不在，直接返回false
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
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            #print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            #print(point12lng)
            # 点在多边形边上
            if (point12lng == point[0]):
                #print("点在多边形边上")
                return False
            if (point12lng > point[0]):
                count +=1
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
    center_x = 0
    center_y = 0
    area = 0.0

    for i in range(len(p) - 1):
        area = area + (p[i][0] * p[i + 1][1] - p[i + 1][0] * p[i][1]) / 2
        center_x = center_x + (p[i][0] * p[i + 1][1] - p[i + 1][0] * p[i][1]) * (p[i][0] + p[i + 1][0])
        center_y = center_y + (p[i][0] * p[i + 1][1] - p[i + 1][0] * p[i][1]) * (p[i][1] + p[i + 1][1])

    n = len(p)
    area = area + (p[n - 1][0] * p[0][1] - p[0][0] * p[n - 1][1]) / 2
    center_x = center_x + (p[n - 1][0] * p[0][1] - p[0][0] * p[n - 1][1]) * (p[n - 1][0] + p[0][0])
    center_y = center_y + (p[n - 1][0] * p[0][1] - p[0][0] * p[n - 1][1]) * (p[n - 1][1] + p[0][1])
    
    center_x /= 6*area
    center_y /= 6*area

    #print(center_x, center_y)
    return [center_x, center_y]

def cmp(a, b, center):
    if a[0] >= 0 and b[0] < 0:
        return False
    if a[0] == 0 and b[0] == 0:
        return a[1] <= b[1]
    
    det = int((a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1]))
    if det < 0:
        return False
    if det > 0:
        return True
    d1 = int((a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1]))
    d2 = int((b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1]))
    return d1 <= d2

def partition(arr, low, high, center): 
    i = (low-1)         # 最小元素索引
    pivot = arr[high]     
  
    for j in range(low , high): 
  
        # 当前元素小于或等于 pivot
        if not cmp(arr[j], pivot, center): 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return (i+1) 

def mySort(arr, low, high, center):
    if low < high: 
        pi = partition(arr,low,high,center) 
  
        mySort(arr, low, pi-1, center) 
        mySort(arr, pi+1, high, center)


def getFaceAlignment(input):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    return fa.get_landmarks(input)

def getBlendedImg(M, background, foreground, output_path):
    (h, w, c) = background.shape

    for y in range(h):
        for x in range(w):
            # eqn(1)
            background[y, x, 0] = M[y, x] * int(foreground[y][x][0]) + (1 - M[y, x]) * int(background[y][x][0])
            background[y, x, 1] = M[y, x] * int(foreground[y][x][1]) + (1 - M[y, x]) * int(background[y][x][1])
            background[y, x, 2] = M[y, x] * int(foreground[y][x][2]) + (1 - M[y, x]) * int(background[y][x][2])

    #io.imshow(background)
    #io.show()
    io.imsave(output_path, background)

def getFaceAlignment(input):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    return fa.get_landmarks(input)

def calEuclDis(v1, v2):
    if len(v1) != len(v2):
        print("len(v1) not equal to len(v2), calEulDis error")
        return -1
    
    ans = 0.0
    for i in range(len(v1)):
        ans += math.sqrt((v1[i][0] - v2[i][0]) * (v1[i][0] - v2[i][0]) + (v1[i][1] - v2[i][1]) * (v1[i][1] - v2[i][1]))

    return ans

def nearestSearch(b_key_points, path, b_h, b_w, b_c):
    img_names = os.listdir(path)
    min_dis = 10000000.0
    tar_img_path = ''
    for img_name in img_names:
        if not img_name.endswith('jpg') and not img_name.endswith('png') and not img_name.endswith('jpeg'):
            continue
        #print(img_name)
        full_path = path + '/' + img_name
        #print(full_path)
        img = io.imread(full_path)

        f_h = 0
        f_w = 0
        f_c = 0
        (f_h, f_w, f_c) = img.shape
        #print(f_h, f_w, f_c)
        if f_h != b_h or f_w != b_w or f_c != b_c:
            continue

        key_points = getFaceAlignment(img)

        dis = calEuclDis(b_key_points, key_points[0])
        #break
        #print(dis)
        if dis < min_dis:
            min_dis = dis
            tar_img_path = full_path
    
    return tar_img_path

if __name__ == "__main__":
    start_time = time.time()
    #background_img_path = './img/background.png'
    # background image
    background_img_path = sys.argv[1]
    #img_path =  './dfdc_train_part_2'
    # image database for searching
    img_path =  sys.argv[2]
    # output path to save result
    output_path = sys.argv[3]
    h = 0
    w = 0
    c = 0

    background = io.imread(background_img_path)
    mask = io.imread(background_img_path)
    #io.imshow(background)
    #io.show()
    (h, w, c) = background.shape

    key_points = getFaceAlignment(background) # 一维为关键点坐标，二维为type
    #print(key_points[0])
    foreground_img_path = nearestSearch(key_points[0], img_path, h, w, c)
    foreground = io.imread(foreground_img_path)
    print(foreground_img_path)
    #凸包算法
    convex_hull_boundary = force(list(key_points[0]), len(key_points[0]))
    for p in convex_hull_boundary:
        p[1] = h - p[1]
    #print(convex_hull_boundary)
    #convex_hull_boundary = [[100, 100], [110, 100], [110, 120], [100, 120]]
    center = getGravity(convex_hull_boundary)
    mySort(convex_hull_boundary, 0, len(convex_hull_boundary) - 1, center)
    #print(convex_hull_boundary)
    #boundarySort(convex_hull_boundary)
    #showFig(convex_hull_boundary)
    
    M = numpy.zeros((h, w))
    #print(M)
    for y in range(h): # row
        for x in range(w): # col
            if isPointinPolygon([x, y], convex_hull_boundary):
                # 按照paper中用平均值作为灰度值
                M[y, x] = (int(background[y][x][0]) + int(background[y][x][1]) + int(background[y][x][2])) / float(3.0)
                M[y, x] = M[y, x] / 255.0
                #print(M[y, x])
                mask[y, x: ] = 255
            else:
                mask[y, x: ] = 0

    B = numpy.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # eqn(2)
            B[i, j] = 4 * M[i, j] * (1 - M[i, j])
            #print(B[i, j])

    #print(B)
    #io.imshow(mask)
    #io.show()
    # eqn(1)
    getBlendedImg(M, background, foreground, output_path)
    
    end_time = time.time()
    print('At cost: ', end_time - start_time)