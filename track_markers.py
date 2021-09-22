import argparse

import cv2
from falatra.markers import MarkerDetection, MarkersTracker, TrackingFailedError


def main(argv):

    markers = MarkerDetection()
    markers.load(argv.xmlfile)
    tracker = MarkersTracker(markers)

    for imgpath in sorted(argv.images):
        
        image = cv2.imread(imgpath)
        if image is None:
            print('Failed - could not read image {}'.format(imgpath))
            break

        try:
            tracker.update(image, imgpath)
        except TrackingFailedError:
            print('Failed for image {}'.format(imgpath))
            continue
            

        if argv.outdir is not None:
            tracker.save(argv.outdir)
        print('Processed image {}'.format(imgpath))

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default=None,
            help='output directory for labels')
    parser.add_argument('xmlfile', type=str,
            help='XML file defining landmarks')
    parser.add_argument('images', type=str, nargs='+',
            help='Images')
    main(parser.parse_args())

