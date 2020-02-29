import os
from skimage import io
from get_face_alignment import getFaceAlignment
from utils import calEuclDis
import const

def nearestSearch(b_key_points, path, b_h, b_w, b_c, b_name):
    img_names = os.listdir(path)
    min_dis = const.MAX_VAL
    lower_bound = const.SEARCH_LOWER_BOUND
    tar_img_path = ''
    for img_name in img_names:
        if img_name == b_name:
            continue

        if not img_name.endswith('jpg') and not img_name.endswith('png') and not img_name.endswith('jpeg'):
            continue
        #print(img_name)
        full_path = path + img_name
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
        #print(dis)
        if dis < min_dis:
            min_dis = dis
            tar_img_path = full_path

        if dis <= lower_bound:
            tar_img_path = full_path
            break
    
    return tar_img_path