import pickle 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def deserialize_headmodel(serfile):

    with open(serfile, 'rb') as fp:
        model = pickle.load(fp)

    return model


class HeadModel3D(object):
    
    def __init__(self):

        self._keypoints  = []
        self._descriptors = []
        self._landmarks  = {}
    
    def __assertPt3f(self, pt3f):
        if type(pt3f) is np.array:
            if pt3f.shape != (3):
                raise ValueError('Expected pt3f to be a real 3d coordinant')
        else:
            ValueError('pt3f must be a numpy array')

    def setLandmark(self, name, pt3f):
        self.__assertPt3f(pt3f)            
        self._landmarks[name] = pt3f

    def addFeaturePoint(self, pt3f, desc):
        self.__assertPt3f(pt3f)
        self._keypoints.append(pt3f)
        self._descriptors.append(desc)
        
    def serialize(self, filename):
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def display(self, selectedlandmarks=None):
        """3D Display using matplotlib 

        Args:
            selectedlandmarks ([type], optional): list of landmarks to display. Defaults to None which shows all landmarks.
        """
       
        kps3d = self.keypoints
        fig = plt.figure() 
        ax = Axes3D(fig)
        ax.scatter(kps3d[:,0], kps3d[:,1], kps3d[:,2],
                marker='o', color='red', s=5, label='Keypoints')

        if len(self._landmarks) > 0:
            landmarks3d = np.array(list(self._landmarks.values()))
            ax.scatter(landmarks3d[:,0], landmarks3d[:,1], landmarks3d[:,2],
                    marker='o', color='blue', s=30, label='Landmarks')
            for name, pt3d in self._landmarks.items():
                pt3d = np.array(pt3d).flatten()
                ax.text(*pt3d, name, color='blue', fontsize='xx-small')
        plt.show()

    @property
    def keypoints(self):
        return np.array(self._keypoints)

    @keypoints.setter
    def keypoints(self):
        raise AttributeError('Cannot modify this field')

    @property
    def descriptors(self):
        return np.array(self._descriptors)

    @descriptors.setter
    def descriptors(self):
        raise AttributeError('Cannot modify this field')

