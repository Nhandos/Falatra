#!/usr/bin/env python3

""" A simple script to load keypoints and images and then perform Brute-Force matching """

import argparse
import logging
import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import cv2

MIN_MATCH_COUNT = 10

normtype = {
    'L2SQR': cv2.NORM_L2SQR,
    'L2': cv2.NORM_L2,
    'L1': cv2.NORM_L1}

root = logging.getLogger()
root.setLevel(logging.DEBUG)

# Suppress matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def _main(_args):
       
    if _args.output:
        assert os.path.isdir(_args.output), "output directory does not exists."

    detector = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10
    )
    logging.info('reference: {}'.format(_args.reference))
    trainImg = cv2.imread(_args.reference)
    trainKP, trainDes = detector.detectAndCompute(trainImg, None)
    bf = cv2.BFMatcher(normtype[_args.normtype], crossCheck=False)

    # Show reference keypoitns
    plt.figure(_args.reference)
    vis = cv2.drawKeypoints(trainImg, trainKP, None)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    if _args.output is not None:
        _, fn = os.path.split(_args.reference)
        path = os.path.join(_args.output, ''.join(fn.split('.')[:-1]) + '-REF.jpg')
        cv2.imwrite(path, vis)
        logging.info('saved reference KP to {}'.format(path))

    for imgPath in _args.image:

        logging.info('query: {}'.format(imgPath))
        queryImg = cv2.imread(imgPath)
        queryKP, queryDes = detector.detectAndCompute(queryImg, None)
        matches = bf.knnMatch(trainDes, queryDes, 2) # requires atleast k = 2 for loweRatioFilter
        loweratio = 0.7
        good = []
        for i, knnmatch in enumerate(matches):
            m, n = knnmatch[:2]
            if m.distance < loweratio * n.distance:
                good.append([m])

        if len(good) > MIN_MATCH_COUNT:

            # draw good matches
            vis = cv2.drawMatchesKnn(trainImg, trainKP, queryImg, queryKP, good, None, singlePointColor=(0, 0, 255))
            
            if _args.output is not None:
                _, fn = os.path.split(imgPath)
                path = os.path.join(_args.output, ''.join(fn.split('.')[:-1]) + '-MATCH.jpg')
                cv2.imwrite(path, vis)
                logging.info('saved query KP to {}'.format(path))

            plt.figure(imgPath)
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        else:
            logging.info('no good KP detected')
    
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Keypoints BF Matcher")
    parser.add_argument('reference', type=str, 
        help='reference image')
    parser.add_argument('image', type=str, nargs='+',  
        help='Images to compare against the reference image')
    parser.add_argument('-e', '--extractor', type=str, nargs=1, default='SIFT',
        help='Keypoint extractor algorithm, default = SIFT')
    parser.add_argument('-nt', '--normtype', type=str, default='L2',
        help='Distance normalisation for computing matches')
    parser.add_argument('-c', '--config', type=str, nargs=1, default='./extractor.conf.json',
        help='Specify configuration file for extractor settings') 
    parser.add_argument('--output', type=str, default=None,
        help='specify directory to output file')
    
    args = parser.parse_args()
    _main(args)