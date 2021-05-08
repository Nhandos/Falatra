import pickle

with open('./data/headmodelfront.ser', 'rb') as fp:
    headmodel = pickle.load(fp)


headmodel.display()
headmodel.display3D()
