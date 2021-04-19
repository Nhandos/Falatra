import json

from falatra.model.stereo import StereoCameraModel

def test():

    CALIBRATION_FILE = './data/calibration/stereo.json'

    print('test_loadstereomodel')
    stereomodel = StereoCameraModel()

    with open(CALIBRATION_FILE, 'r') as fp:
        data = json.load(fp)
        stereomodel.loadFromDict(data)

    print(stereomodel)

if __name__ == '__main__':
    test()

