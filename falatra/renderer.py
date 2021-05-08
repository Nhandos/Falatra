import os
import vtk


def importOBJModel(modelfile, texturepath=None):

    importer = vtk.vtkOBJImporter()
    importer.SetFileName(modelfile)
    if texturepath:
        path, _ = os.path.split(texturepath)
        importer.SetFileNameMTL(texturepath)
        importer.SetTexturePath(path)

    renderer = vtk.vtkRenderer()
    rendererwin = vtk.vtkRenderWindow()
    rendererwin.AddRenderer(renderer)
    importer.SetRenderWindow(rendererwin)
    importer.Update()

    return renderer 
