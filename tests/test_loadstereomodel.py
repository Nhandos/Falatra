import json
import cv2
import numpy as np
from matplotlib import pyplot as plt

from falatra.model.stereo import StereoCameraModel

CALIBRATION_FILE='./data/calibration/stereo2.json'
RECTIFICATION_FILE=('./data/calibration_images/centre/frame_249.png',
        './data/calibration_images/left/frame_249.png')

stereomodel = None

def testInit():
    global stereomodel

    print('Test loading stereomodel...', end=' ')
    stereomodel = StereoCameraModel()
    try:
        with open(CALIBRATION_FILE, 'r') as fp:
            data = json.load(fp)
            stereomodel.loadFromDict(data)
        
        print('Ok.')
        print(stereomodel)
    except Exception as e:
        print('FAILED! - ', e)


def testRectify():
    global stereomodel

    frame1, frame2 = map(cv2.imread, RECTIFICATION_FILE)

    print('Test stereo rectify...', end=' ')
    try:
        stereomodel.initUndistortRectifyMap()
        outframe1, outframe2 = stereomodel.rectify(frame1, frame2)
        print('Ok.')
        plt.figure()
        plt.imshow(np.hstack((outframe1, outframe2)))
        plt.show()

    except Exception as e:
        print('FAILED! - ', e)
        raise e


if __name__ == '__main__':
    testInit()
    testRectify()

