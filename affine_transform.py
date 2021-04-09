import torch
from PIL import Image
import numpy as np
from mtcnn import *
from visualization_utils import *
import cv2 as cv

'''
来源： https://www.pythonf.cn/read/91991
该代码实现利用人脸的五点仿射变换实现人脸对齐
具体就是首先使用mtcnn检测算法检测出人脸区域，并得到lanmarks关键点坐标和检测框坐标
之后对人脸区域外扩60%，然后对该外扩后的区域重新得到关键点，进行五点仿射变换得到即可。
参考链接：https://blog.csdn.net/oTengYue/article/details/79278572
'''

# 最终的人脸对齐图像尺寸分为两种：112x96和112x112，并分别对应结果图像中的两组仿射变换目标点,如下所示
imgSize1 = [112,96]
imgSize2 = [112,112]
# 112x96的目标点
coord5point112x96 = [
    [30.2946, 51.6963],
    [65.5318, 51.6963],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.3655]
]
# 112x112的目标点
coord5point112x112 = [
    [30.2946+8.0000, 51.6963],
    [65.5318+8.0000, 51.6963],
    [48.0252+8.0000, 71.7366],
    [33.5493+8.0000, 92.3655],
    [62.7299+8.0000, 92.3655]
]

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst

def alignment(image, bounding_boxes, landmarks):
    '''
    人脸校准
    Arguments:
        image: PIL图像
    '''
    np_img = np.asarray(image, dtype=np.float32)
    height, width = np_img.shape[:2]
    face_size = bounding_boxes.shape[0]
    faces = []
    for i in range(face_size):
        img = np_img.copy()
        x1, y1, x2, y2 = bounding_boxes[i][:4]
        # 外扩大100%，防止对齐后人脸出现黑边
        new_x1 = max(int(1.50 * x1 - 0.50 * x2),0)
        new_x2 = min(int(1.50 * x2 - 0.50 * x1),width-1)
        new_y1 = max(int(1.50 * y1 - 0.50 * y2),0)
        new_y2 = min(int(1.50 * y2 - 0.50 * y1),height-1)
        # 得到原始图中关键点坐标
        left_eye_x, right_eye_x, nose_x, left_mouth_x, right_mouth_x = landmarks[i,:5]
        left_eye_y, right_eye_y, nose_y, left_mouth_y, right_mouth_y = landmarks[i,5:]
        # 得到外扩100%后图中关键点坐标
        left_eye_x, right_eye_x, nose_x, left_mouth_x, right_mouth_x = np.array([left_eye_x, right_eye_x, nose_x, left_mouth_x, right_mouth_x]) - new_x1
        left_eye_y, right_eye_y, nose_y, left_mouth_y, right_mouth_y = np.array([left_eye_y, right_eye_y, nose_y, left_mouth_y, right_mouth_y]) - new_y1
        # 在扩大100%人脸图中关键点坐标
        face_landmarks = [
            [left_eye_x,left_eye_y],
            [right_eye_x,right_eye_y],
            [nose_x,nose_y],
            [left_mouth_x,left_mouth_y],
            [right_mouth_x,right_mouth_y]
        ]
        # 扩大100%的人脸区域
        face = cv.cvtColor(img[new_y1: new_y2, new_x1: new_x2], cv.COLOR_RGB2BGR)  # PIL Image采用的是RGB格式，所以这里需要转换
        dst = warp_im(face,face_landmarks,coord5point112x112)
        crop_im = dst[0:imgSize2[0],0:imgSize2[1]]
        ret_img = np.floor(cv.cvtColor(crop_im, cv.COLOR_BGR2RGB)).astype(np.uint8)
        faces.append(ret_img)
    return faces
