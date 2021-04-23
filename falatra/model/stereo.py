import argparse 
import os
import json

from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser() 
parser.add_argument('folder1', type=str) 
parser.add_argument('folder2', type=str)
parser.add_argument('imagesize', type=int, nargs=2)
parser.add_argument('--load', type=str, default=None)


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


class CameraModel(object):

    def __init__(self, image_size=None): 
        self.image_size       = image_size
        self.distortion       = None
        self.intrinsic        = None

    def __str__(self):
        separator = "=" * 70
        str_ = '\n'.join([separator,
                          "<Image Size>",
                          str(self.image_size),
                          "\n<Intrinsics>",            
                          str(self.intrinsic),
                          "\n<Distortion Coefficients>",
                          str(self.distortion),
                          separator])

        return str_

    def calibrate(self, obj_pts, img_pts):
        ret, intrinsic, distortion, _, _ = cv2.calibrateCamera(
                obj_pts,
                img_pts,
                self.image_size,
                None, None)

        if ret:
            self.intrinsic  = intrinsic
            self.distortion = distortion

        return ret

    def exportToDict(self):
        """ Export parameters as a Dictionary """
        return {
            'imagesize': self.image_size,
            'intrinsic': self.intrinsic.tolist(),
            'distortion': self.distortion.tolist()
        }

    def loadFromDict(self, valuedict):
        """ Import parameters from dictionary """
 
        self.image_size, self.intrinsic, self.distortion = \
                map(lambda x: np.array(x, np.float64), valuedict.values())
        self.image_size = tuple(map(int, self.image_size.tolist()))


class StereoCameraModel(object):

    STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    STEREO_FLAGS = 0
    #STEREO_FLAGS |= cv2.CALIB_FIX_INTRINSIC  

    def __init__(self, image_size=None):

        self.image_size = image_size

        # Camera model
        self.camera1 = CameraModel(image_size)
        self.camera2 = CameraModel(image_size)

        # Stereo parameters
        self.R        = None   # Rotation matrix from camera_1 to camera_2
        self.T        = None   # Translation vector from camera_1 to camera_2
        self.F        = None   # Fundamental matrix
        self.E        = None   # Essential Matrix

        # Stereo Rectification parameters
        self.rect_trans        = [None, None]  # Rectify transform
        self.proj_mats         = [None, None]  # Projection matrix
        self.disp_to_depth_map = None
        self.valid_boxes       = [None, None]
        self.undistort_map     = [None, None]
        self.rectification_map = [None, None]

    def __str__(self):
        
        separator = "=" * 70
        str_ = '\n'.join([separator,
                          "<Dimension>",
                          str(self.image_size),
                          "\n<Camera1 Intrinsic>",            
                          str(self.camera1.intrinsic),
                          "\n<Camera1 Distortion Coefficients>",
                          str(self.camera1.distortion),
                          "\n<Camera2 Intrinsic>",
                          str(self.camera2.intrinsic),
                          "\n<Camera2 Distortion Coefficients>",
                          str(self.camera2.distortion),
                          "\n<Rotation Matrix>",
                          str(self.R),
                          "\n<Translation Vector>",
                          str(self.T),
                          "\n<Essential Matrix>",
                          str(self.E),
                          "\n<Fundamental Matrix>",
                          str(self.F),
                          separator,
                          ])
        return str_

    def calibrate(self, obj_pts, cam1_pts, cam2_pts):

        ret = self.camera1.calibrate(obj_pts, cam1_pts)  # Camera1 calibration

        if not ret:
            return False

        ret = self.camera2.calibrate(obj_pts, cam2_pts)  # Camera2 calibration

        if not ret:
            return False

        ret, cam1_intrinsic, cam1_distortion, cam2_intrinsic, cam2_distortion, \
                R, T, E, F = cv2.stereoCalibrate(
                    obj_pts,
                    cam1_pts,
                    cam2_pts,
                    self.camera1.intrinsic,
                    self.camera1.distortion,
                    self.camera2.intrinsic,
                    self.camera2.distortion,
                    self.image_size,
                    self.R,
                    self.T,
                    self.E,
                    self.F,
                    criteria=self.STEREO_CRITERIA,
                    flags=self.STEREO_FLAGS)

        if ret:
            self.R, self.T, self.E, self.F = R, T, E, F
            self.camera1.intrinsic = cam1_intrinsic
            self.camera1.distortion = cam1_distortion
            self.camera2.intrinsic = cam2_intrinsic
            self.camera2.distortion = cam2_distortion


        else:
            return False

        return True

    def initUndistortRectifyMap(self):

        (self.rect_trans[0], self.rect_trans[1],
        self.proj_mats[0], self.proj_mats[1],
        self.disp_to_depth_mat, self.valid_boxes[0],
        self.valid_boxes[1]) = cv2.stereoRectify(
            self.camera1.intrinsic,
            self.camera1.distortion,
            self.camera2.intrinsic,
            self.camera2.distortion,
            self.image_size,
            self.R,
            self.T,
            flags=0)

        for i in range(2):
            camera = self.camera1 if i == 0 else self.camera2
            (self.undistort_map[i],
                self.rectification_map[i]) = cv2.initUndistortRectifyMap(
                    camera.intrinsic,
                    camera.distortion,
                    self.rect_trans[i],
                    self.proj_mats[i],
                    self.image_size,
                    cv2.CV_32FC1)

        print(self.rect_trans, self.proj_mats)

    def rectify(self, frame1, frame2):

        new_frames = []
        for i, frame in enumerate((frame1, frame2)):
            new_frames.append(cv2.remap(frame,
                self.undistort_map[i],
                self.rectification_map[i],
                cv2.INTER_NEAREST))

        return new_frames
        

                
    def exportToDict(self):

        return {
            'imagesize': self.image_size,
            'camera1': self.camera1.exportToDict(),
            'camera2': self.camera2.exportToDict(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist()
        }

    def loadFromDict(self, valuedict):

        self.image_size = tuple(map(int, valuedict['imagesize']))
        self.camera1.loadFromDict(valuedict['camera1'])
        self.camera2.loadFromDict(valuedict['camera2'])
        self.R, self.T, self.E, self.F = \
            map(lambda x: np.array(x, np.float64), list(valuedict.values())[3:])


class StereoCalibrator(object):

    POINTS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    def __init__(self, image_size=None):
        
        self.image_size  = image_size 
        self.stereomodel = StereoCameraModel(image_size)
        self.obj_pts     = []
        self.img_pts     = [[], []]
        self.images      = [[], []]

    def __str__(self):
        str_ = ""
        str_ += self.stereomodel.__str__()

        return str_

    def findCalibrationPoints(self, img1, img2, chessboard_size=(7,9)):

        if self.image_size is None:
            self.image_size = img1.shape[:2]
        
        if not (img1.shape[:2] == img2.shape[:2] == self.image_size):
            print("Image size mismatch")
            return

        points = np.zeros((np.prod(chessboard_size), 3), np.float32)
        points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)

        ret1, pts_1 = cv2.findChessboardCorners(img1, chessboard_size, None)
        ret2, pts_2 = cv2.findChessboardCorners(img2, chessboard_size, None)

        if ret1 and ret2:
            pts_1 = cv2.cornerSubPix(img1, pts_1, (11, 11), (-1, -1), self.POINTS_CRITERIA)
            pts_2 = cv2.cornerSubPix(img1, pts_2, (11, 11), (-1, -1), self.POINTS_CRITERIA)

            self.obj_pts.append(points)
            self.img_pts[1].append(pts_1)
            self.img_pts[2].append(pts_2)
            self.images[1].append(img1)
            self.imges[2].append(img2)

            print("Found calibration points in image pair")
        else:
            print("Cannot find calibration points in image pair")

    def addCalibrationPoints(self, img1, img2, world_pts, img_pts1, img_pts2):

        self.obj_pts.append(world_pts)
        self.img_pts[0].append(img_pts1)
        self.img_pts[1].append(img_pts2)
        self.images[0].append(img1)
        self.images[1].append(img2)

    def calibrate(self):
        print('Calibrating cameras...', end=' ')
        ret = self.stereomodel.calibrate(self.obj_pts, self.img_pts[0], self.img_pts[1])
        if ret:
            print('Ok.')
        else:
            print('FAILED!')

        return ret

    def drawChessBoardCorners(self,index, chessboard_size=(7,9)):
        img1 = self.images[0][index]
        img2 = self.images[1][index]
        cv2.drawChessboardCorners(img1, chessboard_size, self.img_pts[0][index], True)
        cv2.drawChessboardCorners(img2, chessboard_size, self.img_pts[1][index], True)
        vis = cv2.resize(np.hstack((img1, img2)), None, fx=0.5, fy=0.5)

        return vis

    def save(self, path):

        print("Saving calibration parameters...", end=' ')
        folder, name = os.path.split(path)
        if os.path.isdir(folder):
            with open(path, 'w+') as fp:
                json.dump(self.stereomodel.exportToDict(), fp)
            print('Ok.')
        else:
            print("Failed! Directory does not exists")
         

