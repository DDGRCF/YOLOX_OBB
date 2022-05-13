import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from yolox.ops import multiclass_obb_nms
from functools import partial
from tqdm.contrib.concurrent import process_map


# def function(m, prog=None, lock=None):
#     b, s = m
#     lock.acquire()
#     prog.value += 1
#     print(prog.value)
#     lock.release()


#     return multiclass_obb_nms(b, s, iou_thr=0.1)
def function(m):
    b, s = m
    return multiclass_obb_nms(b, s, iou_thr=0.1)

if __name__ == "__main__":
    np_boxes = np.array([[6.0, 3.0, 8.0, 7.0, 0.], [3.0, 6.0, 9.0, 11.0, 0.],
                        [3.0, 7.0, 10.0, 12.0, 0.], [1.0, 4.0, 13.0, 7.0, 0.]],
                    dtype=np.float32)
    np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    a = [np_boxes] * 100
    b = [np_scores] * 100
    # boxes = torch.from_numpy(np_boxes)
    # scores = torch.from_numpy(np_scores)  
    # pool = mp.Pool(10)
    # manager = mp.Manager()
    # warp_function = partial(function, prog=manager.Value("i", 0), lock=manager.Lock())
    # results = pool.map(warp_function, zip(a, b))


    results = process_map(function, zip(a, b), max_workers=10, chunksize=10)

    print(results)


