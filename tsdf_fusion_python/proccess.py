#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import time
import sys
import glob
import os
import shutil
from transforms3d import quaternions

path = './capture/2'

# 图片文件复制与标准化
# imagelistcolor= sorted(glob.glob(os.path.join(path,'*.jpg')))
# for image in imagelistcolor:
#     basename = os.path.basename(image)
#     dirname = os.path.dirname(image)

#     dstname = "data/frame-%06d.color.jpg" % int(basename[:2])
#     print dstname
#     # dstname = os.path.join(basename,'*.jpg')
#     # print basename, dirname
#     shutil.copyfile(image, dstname)

# imagelistdepth= sorted(glob.glob(os.path.join(path,'*.png')))
# for image in imagelistdepth:
#     basename = os.path.basename(image)
#     dirname = os.path.dirname(image)

#     dstname = "data/frame-%06d.depth.png" % int(basename[:2])
#     print basename
#     # dstname = os.path.join(basename,'*.jpg')
#     # print basename, dirname
#     shutil.copyfile(image, dstname)

# 获取姿态，生成姿态文件
fi = open(path + "/pose", "r")
posflag = False
oriflag = False
num = 0
for line in fi:
    if "position" in line:
        posflag = True
        continue
    if "orientation" in line:
        oriflag = True
        continue

    # 获取position
    if posflag:
        line = line.strip("\n").strip(" ")[1:-1].split(",")
        posex, posey, posez = float(line[0]), float(line[1]), float(line[2])
        print "p", num, line, posex, posey, posez
        posflag = False

    # 获取orientation
    if oriflag:
        oriflag = False
        line = line.strip("\n").strip(" ")[1:-1].split(",")
        orix, oriy, oriz, oriw = float(line[0]), float(line[1]), float(line[2]), float(line[3])
        print "o", num, line, orix, oriy, oriz, oriw

        quat_wxyz = (oriw, orix, oriy, oriz)
        rotation_matrix = quaternions.quat2mat(quat_wxyz)
        print rotation_matrix

        matrix = np.zeros((4, 4))

        matrix[:3,:3] = rotation_matrix
        matrix[:3,3] = np.array([posex, posey, posez]).T
        matrix[3][3] = 1.0
        print matrix
        np.savetxt("data/frame-%06d.pose.txt" % num, matrix)

        # 每次读完orientation进行处理

        num += 1


