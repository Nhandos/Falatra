import cv2
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def create_points_glyph3DMapper(worldPts):
        mapPoints = vtk.vtkPoints()
        for i, worldPt in enumerate(worldPts):
            mapPoints.InsertNextPoint(*worldPt)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(mapPoints)

        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(1.0)

        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetSourceConnection(sphereSource.GetOutputPort())
        mapper.SetInputData(polydata)

        return mapper


def pickPoints(renderer: vtk.vtkRenderer, points, imagesize):


    pickedPts = []
    picker = vtk.vtkPointPicker()

    for x, y in points:

        nx = x
        ny = -y + imagesize[0]


        picker.Pick(nx, ny, 0, renderer)
        worldPos = picker.GetPickPosition()
        pickedPts.append(worldPos)

    return pickedPts


def takeScreenshot(renwin: vtk.vtkRenderWindow, scale: int=1):
    """ Returns a image in the form of a numpy array of the 
        renderedWindow
    """

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renwin)
    windowToImageFilter.SetScale(scale)
    windowToImageFilter.SetInputBufferTypeToRGB()
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()

    image = windowToImageFilter.GetOutput()

    cols, rows, _ = image.GetDimensions()
    vtkArr = image.GetPointData().GetScalars()
    components = vtkArr.GetNumberOfComponents()

    img = vtk_to_numpy(vtkArr).reshape(rows, cols, components)
    img = np.flip(img, axis=0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

