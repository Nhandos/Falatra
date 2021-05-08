from typing import Tuple

import cv2
import numpy as np

import falatra.utils


class FaceDetector(object):

    def __init__(self):
        self.face_cascade = \
                cv2.CascadeClassifier('./res/haarcascade_frontalface_alt.xml')

    def __call__(self, image: np.ndarray):

        if not falatra.utils.isGrayScale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face = list(self.face_cascade.detectMultiScale(image, 1.3, 5))

        if len(face) == 0:
            return None

        face.sort(key = lambda bbox: bbox[2] * bbox[3])  # sort by area
        return face[-1]


class KeypointsDetector(object):

    def  __init__(self):

        self.detector = cv2.xfeatures2d.SIFT_create()
    
    def __call__(self, image: np.ndarray, roi=None):

        if roi is not None:
            mask = falatra.utils.maskFromBbox(roi, image.shape[:2])
        else:
            mask = None

        if not falatra.utils.isGrayScale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # feature detection + extraction
        pts = cv2.goodFeaturesToTrack(image, 3000,
                qualityLevel=0.001, minDistance=7, mask=mask)

        if pts is None:
            print('not points detected\n')
            return None, None

        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        kps, des = self.detector.compute(image, kps)

        return kps, des


class KeyPoint(object):
    """ Keypoint class cloned from opencv keypoint
        but allows pickling
    """

    def __init__(self,
            pt: Tuple[float, float],
            size: float,
            angle: float,
            response: float,
            octave: int,
            class_id: int):

        self.pt = pt
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id


class Frame(object):


    def __init__(self, image: np.ndarray): 

        self.image = image
        self.roi = None
        self.kps = None
        self.des = None

    def __getstate__(self):

        kps = []
        for kp in self.kps:
            kps.append(KeyPoint(kp.pt, kp.size, kp.angle, kp.response,
                kp.octave, kp.class_id))

        self.kps = kps
        return self.__dict__

    def __setstate__(self, d):

        self.__dict__ = d
        kps = []
        for kp in self.kps:
            kps.append(cv2.KeyPoint(kp.pt, kp.size, kp.angle, kp.response,
                kp.octave, kp.class_id))

        self.kps = kps

    def detect(self):
        
        facedetector = FaceDetector()
        kpdetector = KeypointsDetector()

        self.roi = facedetector(self.image)
        self.kps, self.des = kpdetector(self.image, self.roi)

    def getKeypointsVisual(self):

        ret = np.copy(self.image)

        # ROI
        if self.roi is not None:
            x, y, w, h = self.roi
            ret = cv2.rectangle(ret, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # keypoints labelling
        ret = cv2.drawKeypoints(ret, self.kps, None, color=(0, 0, 255), 
            flags=0)

        return ret

    def getSize(self):
        return self.image.shape[:2]

    @property
    def image(self):
        return self._image


    @image.setter
    def image(self, image):
        assert image is not None
        self._image = image



