import cv2
import torch
import numpy as np
import BboxToolkit as bt
PI = np.pi

def regular_theta(theta, mode='180', start=-PI/2):
    assert mode in ['360', '180']
    cycle = 2 * PI if mode == '360' else PI

    theta = theta - start
    theta = theta % cycle
    return theta + start


def mintheta_obb(rbox):
    x, y, w, h, theta, cls = np.split(rbox, 6, axis=-1)
    theta1 = regular_theta(theta)
    theta2 = regular_theta(theta + PI / 2)
    abs_theta1 = np.abs(theta1)
    abs_theta2 = np.abs(theta2)

    w_regular = np.where(abs_theta1 < abs_theta2, w, h)
    h_regular = np.where(abs_theta1 < abs_theta2, h, w)
    theta_regular = np.where(abs_theta1 < abs_theta2, theta1, theta2)
    # theta_regular = np.where(theta_regular == -PI / 4, PI / 4, theta_regular) 
    return np.concatenate([x, y, w_regular, h_regular, theta_regular, cls], axis=-1)


# def regular_obb(obboxes):
#     x, y, w, h, theta = obboxes.unbind(dim=-1)
#     w_regular = torch.where(w > h, w, h)
#     h_regular = torch.where(w > h, h, w)
#     theta_regular = torch.where(w > h, theta, theta+PI/2)
#     theta_regular = regular_theta(theta_regular)
#     return torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


# def mintheta_obb(obboxes):
#     ''''
#     norm the theta to (-45, 45), and its w, h need to be exchanged
#     '''

#     x, y, w, h, theta = obboxes.unbind(dim=-1)
#     theta1 = regular_theta(theta)
#     theta2 = regular_theta(theta + pi/2)
#     abs_theta1 = torch.abs(theta1)
#     abs_theta2 = torch.abs(theta2)

#     w_regular = torch.where(abs_theta1 < abs_theta2, w, h)
#     h_regular = torch.where(abs_theta1 < abs_theta2, h, w)
#     theta_regular = torch.where(abs_theta1 < abs_theta2, theta1, theta2)

#     obboxes = torch.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)

#     return obboxes
def poly2obb_np(rbox):
    if rbox.shape[-1] == 9:
        res = np.empty((*rbox.shape[:-1], 6))
        res[..., 5] = rbox[..., 8]
        rbox = rbox[..., :8].reshape(1, -1, 2).astype(np.float32)
    elif rbox.shape[-1] == 8:
        res = np.empty([*rbox.shape[:-1], 5])
        rbox = rbox.reshape(1, -1, 2).astype(np.float32) 
    else:
        raise NotImplementedError
    (x, y), (w, h), angle = cv2.minAreaRect(rbox)
    if w >= h:
        angle = -angle
    else:
        w, h = h, w
        angle = -90 - angle
    theta = angle / 180 * PI
    res[..., 0] = x 
    res[..., 1] = y
    res[..., 2] = w
    res[..., 3] = h
    res[..., 4] = theta

    return res[None, :]

def poly2obb(polys):
    polys_np = polys.detach().cpu().numpy()

    order = polys_np.shape[:-1]
    num_points = polys_np.shape[-1] // 2
    polys_np = polys_np.reshape(-1, num_points, 2)
    polys_np = polys_np.astype(np.float32)

    obboxes = []
    for poly in polys_np:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * PI
        obboxes.append([x, y, w, h, theta])

    if not obboxes:
        obboxes = np.zeros((0, 5))
    else:
        obboxes = np.array(obboxes)

    obboxes = obboxes.reshape(*order, 5)
    return polys.new_tensor(obboxes)


# def rectpoly2obb(polys):
#     theta = torch.atan2(-(polys[..., 3] - polys[..., 1]),
#                         polys[..., 2] - polys[..., 0])
#     Cos, Sin = torch.cos(theta), torch.sin(theta)
#     Matrix = torch.stack([Cos, -Sin, Sin, Cos], dim=-1)
#     Matrix = Matrix.view(*Matrix.shape[:-1], 2, 2)

#     x = polys[..., 0::2].mean(-1)
#     y = polys[..., 1::2].mean(-1)
#     center = torch.stack([x, y], dim=-1).unsqueeze(-2)
#     center_polys = polys.view(*polys.shape[:-1], 4, 2) - center
#     rotate_polys = torch.matmul(center_polys, Matrix.transpose(-1, -2))

#     xmin, _ = torch.min(rotate_polys[..., :, 0], dim=-1)
#     xmax, _ = torch.max(rotate_polys[..., :, 0], dim=-1)
#     ymin, _ = torch.min(rotate_polys[..., :, 1], dim=-1)
#     ymax, _ = torch.max(rotate_polys[..., :, 1], dim=-1)
#     w = xmax - xmin
#     h = ymax - ymin

#     obboxes = torch.stack([x, y, w, h, theta], dim=-1)
#     return regular_obb(obboxes)


def poly2hbb(polys):
    polys = polys.view(*polys.shape[:-1], polys.size(-1)//2, 2)
    lt_point = torch.min(polys, dim=-2)[0]
    rb_point = torch.max(polys, dim=-2)[0]
    return torch.cat([lt_point, rb_point], dim=-1)


def obb2poly(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)

    vector1 = torch.cat(
        [w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat(
        [-h/2 * Sin, -h/2 * Cos], dim=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return torch.cat(
        [point1, point2, point3, point4], dim=-1)


def obb2hbb(obboxes, mode="xyxy"):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w/2 * Cos) + torch.abs(h/2 * Sin)
    y_bias = torch.abs(w/2 * Sin) + torch.abs(h/2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    if mode == "xyxy":
        obboxes = torch.cat([center-bias, center+bias], dim=-1) 
    elif mode == "cxcywh":
        obboxes = torch.cat([center, 2 * bias.abs()])
    return obboxes


def hbb2poly(hbboxes):
    l, t, r, b = hbboxes.unbind(-1)
    return torch.stack([l, t, r, t, r, b, l ,b], dim=-1)


def hbb2obb(hbboxes):
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)

    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta-PI/2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


_type_func_map = {
    ('poly', 'obb'): poly2obb,
    ('poly', 'hbb'): poly2hbb,
    ('obb', 'poly'): obb2poly,
    ('obb', 'hbb'): obb2hbb,
    ('hbb', 'poly'): hbb2poly,
    ('hbb', 'obb'): hbb2obb
}

def get_bbox_type(bboxes, with_score=False):
    dim = bboxes.size(-1)
    if with_score:
        dim -= 1

    if dim == 4:
        return 'hbb'
    if dim == 5:
        return 'obb'
    if dim  == 8:
        return 'poly'
    return 'notype'

def bbox2type(bboxes, to_type):
    assert to_type in ['hbb', 'obb', 'poly']

    ori_type = get_bbox_type(bboxes)
    if ori_type == 'notype':
        raise ValueError('Not a bbox type')
    if ori_type == to_type:
        return bboxes
    trans_func = _type_func_map[(ori_type, to_type)]
    return trans_func(bboxes)
