from matplotlib import pyplot as plt
import pickle

with open('./tmp.ser', 'rb') as fp:
    headmodel = pickle.load(fp)

with open('./frame1.ser', 'rb') as fp:
    frame = pickle.load(fp)

headmodel.display()
vis = frame.getKeypointsVisual()
plt.figure()
plt.imshow(vis[..., [2,1,0]])
plt.show()
