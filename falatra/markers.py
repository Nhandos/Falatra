import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pascal_voc_writer import Writer

from .utils import draw_bboxes, is_bbox_overlap


class MarkersDetections(object):

    def __init__(self, image, regions, source=None):
        self.image = image
        self.source = source
        self.msers, self.bboxes = regions

    def clone(self):
        return MarkersDetections(self.image, (self.msers, self.bboxes), 
            self.source)

    def save(self, savepath):
        writer = Writer(self.source, *self.image.shape[:2])

        for i, bbox in enumerate(self.bboxes):
            x, y, w, h = bbox
            writer.addObject(str(i), x, y, x+w, y+h)

        _, filename = os.path.split(self.source)
        filename = filename.split('.')[-2]
        writer.save(os.path.join(savepath, f'{filename}.xml'))

    def display(self):
        plt.figure()
        vis = draw_bboxes(self.image, self.bboxes)       
        
        if len(vis.shape) == 2:
            plt.imshow(vis, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        plt.show()

    def remove_overlap(self):
        """ Remove overlapping MSER regions """

        # sort regions by area
        regions = list(zip(self.msers, self.bboxes))
        regions.sort(key=lambda region: region[1][2] * region[1][3])

        keep_idx = [i for i in range(len(regions))]
        for i in range(len(regions)):
            if keep_idx[i] == -1:  #index has been pruned
                continue  
            _, bbox1 = regions[i]  # bbox to keep
            
            # prune all bbox that's inside 
            for j in range(i+1, len(regions)):
                _, bbox2 = regions[j]
                if is_bbox_overlap(bbox1, bbox2):
                    keep_idx[i] = -1  # mark as pruned

        # copy filtered regions
        msers = []
        bboxes = []
        for i in keep_idx:
            if i == -1:
                continue
            mser, bbox = regions[i]
            msers.append(mser)
            bboxes.append(bbox)

        result = self.clone()
        result.msers = msers
        result.bboxes = bboxes

        return result

class MarkersDetector(object):

    def __init__(self):

        self.detector = cv2.MSER_create(
            _delta=4,
            _min_area=90,
            _max_area=250,
            _max_variation=0.45,
            _min_diversity=.2,
            _max_evolution=200,
            _area_threshold=1.01,
            _min_margin=0.003,
            _edge_blur_size=7
        )

    def detect(self, image: np.ndarray, source=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(9, 9), 0)
        blur = cv2.medianBlur(gauss, 13)
        msers, bboxes = self.detector.detectRegions(blur)
        return MarkersDetections(image, (msers, bboxes), source)

