#!/usr/bin/env python
""" Performs stereo calibration using chessboard """
import argparse
import logging
import sys
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from falatra.model.stereo import StereoCalibrator, importCalibrationPoints

parser = argparse.ArgumentParser() 
parser.add_argument('folder1', type=str, 
    help='directory containing first camera calibration images')
parser.add_argument('folder2', type=str,
    help='directory containing second camera calibration images')
parser.add_argument('output', type=str,
    help='directory to save calibration data')
parser.add_argument('--load', type=str, default=None,
    help='object points and image points from file')


def get_imagesize(imagefile):

    img = cv2.imread(imagefile)
    return img.shape[:2]


def _main(args):

    files1 = os.listdir(args.folder1)
    files2 = os.listdir(args.folder2)
    files1.sort()
    files2.sort()

    image_size = get_imagesize(os.path.join(args.folder1, files1[0]))
    calibrator = StereoCalibrator(image_size)

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
            print(i, file1, file2, calibratorfile)
            world_pts, img_pts1, img_pts2 = importCalibrationPoints(calibratorfile)
            calibrator.addCalibrationPoints(img1, img2, world_pts, img_pts1, img_pts2)
        else:
            calibrator.findCalibrationPoints(img1, img2)

    ret = calibrator.calibrate()
    if ret:
        calibrator.save(args.output)
        
    


if __name__ == '__main__':
    _main(parser.parse_args())

