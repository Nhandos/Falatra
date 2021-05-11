import vtk
from .vtkutils import takeScreenshot, pickPoints
from .keypoints import Frame
from .model.head3d import HeadModel


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

