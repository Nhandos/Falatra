import argparse
import json

import cv2
from matplotlib import pyplot as plt
import vtk

from falatra.modelcreator import create_headmodel
from falatra.renderer import importOBJModel
from falatra.vtkutils import takeScreenshot,create_points_glyph3DMapper
from falatra.keypoints import Frame


class HeadModelInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, output, output2, metadata, modelren, parent=None):
        self.modelrenderer = modelren
        self.parent = parent
        self.output = output
        self.output2 = output2
        self.metadata = metadata
        self.AddObserver("KeyPressEvent", self.keyPressEvent)

        self.keyPointsActor = vtk.vtkActor()
        self.modelrenderer.AddActor(self.keyPointsActor)

    def keyPressEvent(self, obj, event):
        key = self.parent.GetKeySym()
        if key == 'k':
            self.modelrenderer.RemoveActor(self.keyPointsActor)

            headmodel, frame = create_headmodel(self.metadata, self.parent.GetRenderWindow(), self.modelrenderer, None)
            mapper = create_points_glyph3DMapper(headmodel.keypoints)
            self.keyPointsActor.SetMapper(mapper)
            self.keyPointsActor.GetProperty().SetColor(198,214,36)

            self.modelrenderer.AddActor(self.keyPointsActor)
            self.parent.Render()

            headmodel.display()
            frame.serialize(self.output2)
            headmodel.serialize(self.output)

def main(argv):

    colors = vtk.vtkNamedColors()
    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)

    with open(argv.metadatafile, 'r') as fp:
        metadata = json.load(fp)

    renderer = importOBJModel(argv.objfile, argv.texturefile)

    # Create the graphics structure. The renderer renders into the render
    # window. The render window interactor captures mouse events and will
    # perform appropriate camera or actor manipulation depending on the
    # nature of the events.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    renderer.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(1440, 900)
    renWin.SetWindowName('Generate Head model')

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create a custom interactor to handle events produced by a user
    # (keypress, mouse events, etc)
    style = HeadModelInteractorStyle(argv.headmodel_out, argv.frame_out, metadata, renderer, parent=iren)
    iren.SetInteractorStyle(style)

    # This allows the interactor to initalize itself. It has to be
    # called before an event loop.
    iren.Initialize()

    # We'll zoom in a little by accessing the camera and invoking a "Zoom"
    # method on it.
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.5)
    renWin.Render()

    # Start the event loop.
    iren.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('objfile', type=str, help='obj model file')
    parser.add_argument('texturefile', type=str,
        help='model texture file')
    parser.add_argument('metadatafile', type=str, help='face metadata')
    parser.add_argument('headmodel_out', type=str, help='headmodel serialize output')
    parser.add_argument('frame_out', type=str, help='serialize the 2d features that was use to create headmodel')
    main(parser.parse_args())


