import vtk
from .vtkutils import takeScreenshot, pickPoints
from .keypoints import Frame
from .model.head3d import HeadModel3D


def create_headmodel(
        metadata: dict,
        renderwindow: vtk.vtkRenderWindow,
        renderer: vtk.vtkRenderer,
        landmarks
    ):

    image = takeScreenshot(renderwindow)
    frame = Frame(image)
    frame.detect()

    headModel = HeadModel3D()
    pts = [kp.pt for kp in frame.kps]
    worldPts = pickPoints(renderer, pts, frame.getSize())
    for keypoint, descriptor in zip(worldPts, frame.des):
        headModel.addFeaturePoint(keypoint, descriptor)


    # Add landmarks
    landmarks = metadata['FaceModel']['Assessments']['Assessment']['Landmarks']

    for side in ['LeftSide', 'Medial', 'RightSide']:
        for key, value in landmarks[side].items():
            pt3f = list(map(float, value.values()))
            name = '_'.join([side, key])
            headModel.setLandmark(name, pt3f)


    return headModel, frame

