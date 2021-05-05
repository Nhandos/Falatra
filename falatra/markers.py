import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pascal_voc_writer import Writer

from .utils import draw_bboxes, is_bbox_overlap, cvt_bbox_coords


class MarkersError(Exception):
    """ Exceptions for Markers """
    pass


class ImageNotFoundError(MarkersError):
    """ Raised when an image is not found """


class MarkersDetections(object):

    def __init__(self, image, bboxes, source=None):
        self.image = image
        self.source = source
        self.bboxes = bboxes

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

    def load(self, xmlpath):
        
        tree = ET.parse(xmlpath)
        root = tree.getroot()

        imgpath = root.find('path').text
        image = cv2.imread(imgpath)

        if image is None:
            raise ImageNotFoundError('Could not load {}'.format(imgpath))
        else:
            self.image = image
            self.source = imgpath
        
        bboxes = []
        for boxes in root.iter('object'):
            
            xmin = int(boxes.find('bndbox/xmin').text)
            ymin = int(boxes.find('bndbox/ymin').text)
            xmax = int(boxes.find('bndbox/xmax').text)
            ymax = int(boxes.find('bndbox/ymax').text)

            bboxes.append(cvt_bbox_coords((xmin, ymin, xmax, ymax), 
                'XYXY_XYWH'))


        self.bboxes = bboxes

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
        bboxes = sorted(self.bboxes, key=lambda bbox: bbox[2] * bbox[3],
                reverse=True)

        keep_idx = list(range(len(bboxes)))
        for i, bbox in enumerate(bboxes):
            if keep_idx[i] == -1:  #index has been pruned
                continue  

            # prune all bbox that's inside 
            for j in range(i + 1, len(bboxes)):
                bbox2 = bboxes[j]
                if is_bbox_overlap(bbox, bboxes[j]):
                    keep_idx[i] = -1  # mark as pruned

        self.bboxes = []
        for i in keep_idx:
            if i != -1:
                self.bboxes.append(bboxes[i])


class MarkersDetector(object):

    def __init__(self):

        self.detector = cv2.MSER_create(
            _delta=5,
            _min_area=30,
            _max_area=300,
            _max_variation=0.4,
            _min_diversity=.2,
            _max_evolution=200,
            _area_threshold=0.001,
            _min_margin=0.003,
            _edge_blur_size=5
        )

    def detect(self, image: np.ndarray, source=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(9, 9), 0)
        blur = cv2.medianBlur(gray, 9)
        _, bboxes = self.detector.detectRegions(blur)

        return MarkersDetections(image, bboxes, source)

