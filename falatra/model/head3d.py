import pickle 

from matplotlib import pyplot as plt
import numpy as np
import vtk

from .landmarks import Landmarks3D
from ..vtkutils import takeScreenshot, pickPoints
from ..keypoints import Frame


def hasFrame(func):

    def inner(self, *args, **kwargs):
        if self.frame is None:
            raise RuntimeError("Model has no frame - Call updateFrame()")
        else:
            return func(self, *args, **kwargs)

    return inner


def create_headmodel(
        renderwindow: vtk.vtkRenderWindow,
        renderer: vtk.vtkRenderer,
        landmarks
    ):

    image = takeScreenshot(renderwindow)
    frame = Frame(image)
    frame.detect()
    
    pts = [kp.pt for kp in frame.kps]
    worldPts = pickPoints(renderer, pts, frame.getSize())
    headmodel = HeadModel(frame, worldPts, frame.des, landmarks)

    return headmodel


def deserialize_headmodel(serfile):

    with open(serfile, 'rb') as fp:
        model = pickle.load(fp)

    return model


class HeadModel(object):

    def __init__(self, frame, kps, des, landmarks):

        self.frame = frame
        self.keypoints = kps    
        self.descriptor = des   
        self.landmarks = landmarks  

    def load(self, filename):
        with open(filename, "rb") as fp:
            return pickle.load(fp)

    def save(self, filename):
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def display(self):

        vis = self.frame.getKeypointsVisual()
        plt.figure()
        plt.imshow(vis[...,[2,1,0]])
        plt.show()

    def display3D(self):
       
        kps3d = np.array(self.keypoints)
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(kps3d[:,0], kps3d[:,1], kps3d[:,2])
        plt.show()

