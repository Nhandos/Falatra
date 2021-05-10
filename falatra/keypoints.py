from typing import Tuple

import cv2
from matplotlib import pyplot as plt
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


class FeatureMatcher(object):
    
    def __init__(self, normtype=cv2.NORM_L2):
        self.bfmatcher = cv2.BFMatcher(normtype, crossCheck=False)

    def __call__(self, kp1, des1, kp2, des2, loweratio=0.7, homography=True):

        # match kps
        matches = self.bfmatcher.knnMatch(des1, des2, 2) # requires atleast 2 nearest matches

        # loweratio filter
        good = []
        for i, knnmatch in enumerate(matches):
            m, n = knnmatch[:2]
            if m.distance < loweratio * n.distance:
                good.append([m])


        if homography:
            src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            mask = np.ones((len(good), 1), dtype=np.uint8)

        return good, mask 


class FrameMatcher(FeatureMatcher):

    def __init__(self, normtype=cv2.NORM_L2):
        self.bfmatcher = cv2.BFMatcher(normtype, crossCheck=False)

    def __call__(self, frame1, frame2, loweratio=0.95, homography=True):
        kp1, des1 = frame1.kps, frame1.des
        kp2, des2 = frame2.kps, frame2.des

        return super().__call__(kp1, des1, kp2, des2, loweratio, homography)

    @classmethod
    def display(self, frame1, frame2, matches, mask=None):

       image = cv2.drawMatchesKnn(
               frame1.image, frame1.kps,
               frame2.image, frame2.kps,
               matches, None,
               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
               matchesMask=mask)

       plt.figure()
       plt.imshow(image[...,[2,1,0]])
       plt.show()


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
            kps.append(cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response,
                kp.octave, kp.class_id))

        self.kps = kps

    def detect(self, detectFace=True):
        
        if detectFace:
            facedetector = FaceDetector()
            self.roi = facedetector(self.image)
        else:
            self.roi = None

        kpdetector = KeypointsDetector()
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



