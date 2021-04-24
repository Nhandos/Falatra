#!/usr/bin/env python
""" Performs stereo calibration using chessboard """
import argparse
import logging
import sys
import os
import json

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from falatra.model.stereo import StereoCalibrator

parser = argparse.ArgumentParser() 
parser.add_argument('folder1', type=str, 
    help='directory containing first camera calibration images')
parser.add_argument('folder2', type=str,
    help='directory containing second camera calibration images')
parser.add_argument('output', type=str,
    help='directory to save calibration data')
parser.add_argument('--load', type=str, default=None,
    help='object points and image points from file')
parser.add_argument('--show', default=False, action='store_true',
    help='Show calibration corners')


def get_imagesize(imagefile):

    img = cv2.imread(imagefile)
    return img.shape[:2]


def importCalibrationPoints(calibfile):

    world_pts = []
    img_pts1 = []
    img_pts2 = []

    with open(calibfile, "r") as f:
        data = json.load(f)
        for row in data:
            world_pts.append(row['obj_pts'])
            img_pts1.append(row['img_pts1'])
            img_pts2.append(row['img_pts2'])

    world_pts = np.array(world_pts).astype(np.float32)
    world_pts = np.insert(world_pts, 2, 0, axis=1)
            
    return np.array(world_pts).astype(np.float32), \
           np.array(img_pts1).astype(np.float32), \
           np.array(img_pts2).astype(np.float32)


def _main(args):

    files1 = os.listdir(args.folder1)
    files2 = os.listdir(args.folder2)
    files1.sort()
    files2.sort()

    image_size = get_imagesize(os.path.join(args.folder1, files1[0]))
    calibrator = StereoCalibrator(6, 8, 25, image_size)
    if args.load:
        calibratorfiles = os.listdir(args.load)
        calibratorfiles.sort()

    for i, (file1, file2) in enumerate(zip(files1, files2)):
        file1 = os.path.join(args.folder1, file1)
        file2 = os.path.join(args.folder2, file2)
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)

        if args.load:
            calibratorfile = os.path.join(args.load, calibratorfiles[i])
            world_pts, img_pts1, img_pts2 = importCalibrationPoints(calibratorfile)
            corners = (img_pts1, img_pts2)
        else:
            corners = None

        calibrator.add_corners((img1, img2), corners, show_results=args.show)

    calib = calibrator.calibrate_cameras()
    print(calibrator.check_calibration(calib))
    print(calib)
    calib.export(args.output)


if __name__ == '__main__':
    _main(parser.parse_args())

