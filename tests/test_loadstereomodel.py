import json
import cv2
import numpy as np
from matplotlib import pyplot as plt

from falatra.model.stereo import StereoCalibration

CALIBRATION_FILE='./data/calibration/calibration1'
RECTIFICATION_FILE=('./data/calibration_images/centre/frame_249.png',
        './data/calibration_images/left/frame_249.png')

calib = None

def testInit():
    global calib 

    print('Test loading stereomodel...', end=' ')
    calib = StereoCalibration()
    
    try:
        calib.load(CALIBRATION_FILE)
        print('Ok.')
        print(calib)
    except Exception as e:
        print('FAILED! - ', e)


def testRectify():
    global calib 

    frame1, frame2 = map(cv2.imread, RECTIFICATION_FILE)

    print('Test stereo rectify...', end=' ')
    try:
        outframe1, outframe2 = calib.rectify((frame1, frame2))
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

