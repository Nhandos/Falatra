import argparse

import cv2
import pygame
from pygame.locals import DOUBLEBUF

from display import Display2D
from falatra.markers import MarkerDetection, MarkersTracker


def main(argv):

    markers = MarkerDetection()
    markers.load(argv.xmlfile)
    tracker = MarkersTracker(markers)

    for imgpath in sorted(argv.images):
        
        image = cv2.imread(imgpath)
        if image is None:
            print('Failed - could not read image {}'.format(imgpath))
            break

        tracker.update(image, imgpath)
        if argv.outdir is not None:
            tracker.save(argv.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default=None,
            help='output directory for labels')
    parser.add_argument('xmlfile', type=str,
            help='XML file defining landmarks')
    parser.add_argument('images', type=str, nargs='+',
            help='Images')
    main(parser.parse_args())

