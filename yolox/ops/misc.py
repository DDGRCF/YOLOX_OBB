import torch
import BboxToolkit as bt
from yolox.utils.obb_utils import bbox2type
from .pytorch import obb_overlaps

class OBBOverlaps():

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        bboxes1 = bboxes1.float()
        bboxes2 = bboxes2.float()
        assert bboxes2.shape[-1] in [0, 5, 6]
        assert bboxes1.shape[-1] in [0, 5, 6]
        
        if bboxes1.shape[-1] == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.shape[-1] == 6:
            bboxes2 = bboxes2[..., :5]
        return obb_overlaps(bboxes1, bboxes2, mode)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


class PolyOverlaps():
    """2D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        if bboxes1.shape[-1] in [5, 6]:
            if bboxes1.shape[-1] == 6:
                bboxes1 = bboxes1[:, :5]
            bboxes1 = bbox2type(bboxes1, 'poly')
        if bboxes2.shape[-1] in [5, 6]:
            if bboxes2.shape[-1] == 6:
                bboxes2 = bboxes1[:, :5]
            bboxes2 = bbox2type(bboxes1, 'poly')

        assert bboxes1.size(-1) in [0, 8, 9]
        assert bboxes2.size(-1) in [0, 8, 9]
        if bboxes2.size(-1) == 9:
            bboxes2 = bboxes2[..., :8]
        if bboxes1.size(-1) == 9:
            bboxes1 = bboxes1[..., :8]
        with torch.cuda.amp.autocast(enabled=False):
            iou_matrix = bt.bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        return iou_matrix

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

