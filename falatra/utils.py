import cv2
import numpy as np

XYWH_XYXY='xywh'
XYXY_XYWH='xyxy'


class UtilsError(Exception):
    """Base class for other exceptions in utils"""
    pass


class ConversionError(UtilsError):
    """Raised when theres a conversion error"""
    pass


def draw_bboxes(image: np.ndarray, bboxes):
    vis = np.copy(image)
    for bbox in bboxes:
        cv2.rectangle(vis, bbox, (255, 0, 0), 2)

    return vis


def in_bbox(point, bbox):
    """ Checks if a point is within a bounding box """
    cond1 = point[0] >= bbox[0] and point[0] <= bbox[0] + bbox[2]
    cond2 = point[1] >= bbox[1] and point[1] <= bbox[1] + bbox[3]
    return cond1 and cond2


def cvt_bbox_coords(bbox, standard):
    """ converts between (x,y,w,h) and (x1,y1,x2,y2) """

    result = None

    if standard == XYXY_XYWH:
        result = bbox[0], bbox[1], bbox[2] - bbox[1], bbox[3] - bbox[2]
    elif standard == XYWH_XYXY:
        result = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    else:
        raise ConverstionError("{} is an incorrect standard".format(standard))
    
    return result


def is_bbox_overlap(bbox1, bbox2):
    """ Calculate if bbox overlaps """
    
    rec1 = cvt_bbox_coords(bbox1, XYWH_XYXY)
    rec2 = cvt_bbox_coords(bbox2, XYWH_XYXY)

    # check if either retangle is acutally a line
    if (rec1[0] == rec1[2] or rec1[1] == rec1[3] or \
        rec2[0] == rec1[2] or rec2[1] == rec1[2]):
        # the line cannot have positive overlap
        return False

    return not (rec1[2] <= rec2[0] or  # left
                rec1[3] <= rec2[1] or  # bottom
                rec1[0] >= rec2[2] or  # right
                rec1[1] >= rec2[3])    # top
    
        
