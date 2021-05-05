import argparse
import os
import glob

import cv2
from matplotlib import pyplot as plt
import numpy as np
from falatra.markers import MarkersDetector


def load_frames(folder):

    paths = sorted(os.listdir(folder))
    return list(map(lambda x: os.path.join(folder, x), paths))


def main(argv):

    paths = argv.images
    detector = MarkersDetector()
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print('Error - could not read {}'.format(path))
            continue
        
        detection = detector.detect(img, path)
        detection.remove_overlap()
        detection.save(argv.outdir)
        print('detected: ', len(detection.bboxes), ' in ', path)
        if argv.display:
            detection.display()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, nargs='+', help='images')
    parser.add_argument('--outdir', type=str, default='/tmp/',
        help='directory to save output labels')
    parser.add_argument('--display', action='store_true',
        help='display detection')

    argv = parser.parse_args()
    main(argv)

