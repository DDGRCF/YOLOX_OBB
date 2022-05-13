import cv2
import numpy as np

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