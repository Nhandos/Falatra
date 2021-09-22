# Falatra
## Overview

(FA)cial (LA)dmarks (TRA)cking is a 3D Facial landmarks tracking algorithm designed within the context of clinical
outcome measures for speech sound disorders intervention.


## Dependencies

We recommend you have Python-3.6.9 and Pip installed.

Run the following command in the root directory to install dependent python packages:
```
pip install -r requirements.txt
```

## Interactive Notebook

For an interactive notebook of the project, see the link below:

https://mybinder.org/v2/gh/Nhandos/Falatra/5be459efb42a2d2a36f68d25977b7db82970bd1c?filepath=demo.ipynb

## Project Structure

### notebooks

This project contains notebooks for the purpose of demo and testing.

### data/

* ``headmodelfront.ser`` & ``headmodelleft.ser``: Serialised HeadModel3D object. See ``falatra/model/head3d.py``

* ``calibration/`` : serialised camera calibration parameters.

* ``training/``: Training data in the form of stereo images + landmarks ground truth labels.

### falatra/

``falatra`` is a package which contains the main tracking code. Here are a high-level overview of the files:

* ``keypoints.py``: 2D feature detection/matching and contains the ``Frame`` and ``FrameMatcher`` and class  which
  runs the main computation.

* ``markers.py``: Bounding box data for ground-truth landmarks. Reading/Saving bounding box & tracking bounding box
  between frames.

* ``modelcreator.py``: Creates a HeadModel3D object from a VTK render.

* ``renderer.py``: Creates a VTK renderer from an OBJ file.

* ``utils.py``: Various unorganised utility functions

* ``vtkutils.py``: Various unorganised VTK related utility functions.

###


### other miscellaneous files

* /matlab: some matlab scripts

* /falatra: otype implementation

* /test: some unit tests

* /tools: auxillary python scripts

# References

This is a facial landmark tracking algorithm developed as
part of my Final Years Engineering Honours Project.

The algorithm that I am developing will aim to help speech
language pathologist make more informed diagnostic assessment 
for children with speech disorders.


