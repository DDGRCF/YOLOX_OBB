import cv2
import numpy as np
from loguru import logger
from pathlib import Path
from shapely.geometry import *

def poly2rbox(seg):
    if isinstance(seg, list):
        if len(seg) > 1:
            seg_all_expand = []
            for s in seg:
                seg_all_expand.extend(s)
            seg_all_expand = np.asarray(seg_all_expand,  dtype=np.float32).reshape(1, -1, 2)
            seg = cv2.convexHull(seg_all_expand, returnPoints=True)
        else:
            seg = np.asarray(seg[0], dtype=np.float32).reshape(1, -1, 2)

    (x, y), (w, h), angle = cv2.minAreaRect(seg)
    if w >= h:
        angle = -angle
    else:
        w, h = h, w
        angle = -90 - angle
    theta = angle / 180 * np.pi
    bbox = np.array((x, y, w, h, theta), dtype=np.float32)
    bbox = obb2poly(bbox)
    return bbox.squeeze()

def obb2poly(obboxes):
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.concatenate(
        [w/2 * Cos, -w/2 * Sin], axis=-1)
    vector2 = np.concatenate(
        [-h/2 * Sin, -h/2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return np.concatenate(
        [point1, point2, point3, point4], axis=-1)


def judge_direction(p1, p2, p3):
    f_p = eval('p{}'.format(str([p1[1], p2[1]].index(max(p1[1], p2[1])) + 1)))
    f_b = eval('p{}'.format(str([p1[1], p2[1]].index(min(p1[1], p2[1])) + 1)))
    s = (f_p[0] - p3[0]) * (f_b[1] - p3[1]) - (f_p[1] - p3[1]) * (f_b[0] - p3[0])
    direction = "left" if s > 0 else "right" if s < 0 else s
    return direction

def get_label(points, objs, temp_label):
    p1 = Polygon(points)
    iou = 0
    max_index = []
    for _, obj in enumerate(objs):
        if obj['label'] != temp_label:
            p2 = Polygon(obj['points'])
            t_iou = p1.intersection(p2).area / p1.union(p2).area
            if t_iou > iou:
                iou = t_iou
                max_index.append(1)
                continue
        max_index.append(0)
    objs[(len(objs) - max_index[::-1].index(1) - 1)]['points'] = points


