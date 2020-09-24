""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import plothand

# replace path to local path to dataset, choose set between training and evaluation set
path = ''
set = 'training'
# set = 'evaluation'

colors = np.array([[0.4, 0.4, 0.4],
                   [0.4, 0.0, 0.0],
                   [0.6, 0.0, 0.0],
                   [0.8, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.4, 0.4, 0.0],
                   [0.6, 0.6, 0.0],
                   [0.8, 0.8, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 0.4, 0.2],
                   [0.0, 0.6, 0.3],
                   [0.0, 0.8, 0.4],
                   [0.0, 1.0, 0.5],
                   [0.0, 0.2, 0.4],
                   [0.0, 0.3, 0.6],
                   [0.0, 0.4, 0.8],
                   [0.0, 0.5, 1.0],
                   [0.4, 0.0, 0.4],
                   [0.6, 0.0, 0.6],
                   [0.7, 0.0, 0.8],
                   [1.0, 0.0, 1.0]])
colors = colors[:, ::-1]
colors = colors * 255
bones = [((0, 4), colors[4, :]),
         ((4, 3), colors[3, :]),
         ((3, 2), colors[2, :]),
         ((2, 1), colors[1, :]),

         ((0, 8), colors[8, :]),
         ((8, 7), colors[7, :]),
         ((7, 6), colors[6, :]),
         ((6, 5), colors[5, :]),

         ((0, 12), colors[12, :]),
         ((12, 11), colors[11, :]),
         ((11, 10), colors[10, :]),
         ((10, 9), colors[9, :]),

         ((0, 16), colors[16, :]),
         ((16, 15), colors[15, :]),
         ((15, 14), colors[14, :]),
         ((14, 13), colors[13, :]),

         ((0, 20), colors[20, :]),
         ((20, 19), colors[19, :]),
         ((19, 18), colors[18, :]),
         ((18, 17), colors[17, :])]


def viz(savefig=False):
    """
     viz is a function that crop and visualize keypoint annotation on image

     Args:
         :param savefile : whether to save the visualized image. default to False

     Returns:
        :return None
    """
    # load annotations of this set
    with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
        anno_all = pickle.load(fi)

    # iterate samples of the set
    itr = 0
    for sample_id, anno in anno_all.items():
        if itr > 3:
            break
        # load data
        image = cv2.imread(os.path.join(set, 'color', '%.5d.png' % sample_id))
        # get info from annotation dictionary
        kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean

        left_hand_uv = kp_coord_uv[0:21]
        right_hand_uv = kp_coord_uv[21:42]
        if np.sum(left_hand_uv[:, 2]) >= 12:
            pts = left_hand_uv.astype('float')
            x_min = min(pts[:, 0])
            x_max = max(pts[:, 0])
            y_min = min(pts[:, 1])
            y_max = max(pts[:, 1])
            x_min = x_min - (x_max - x_min) / 2
            x_max = x_max + (x_max - x_min) / 2
            y_min = y_min - (y_max - y_min) / 2
            y_max = y_max + (y_max - y_min) / 2
            x_min, x_max = max(0, x_min), min(image.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(image.shape[0], y_max)
            img_crop = image[y_min:y_max, x_min:x_max]
            pts[:, 0] = pts[:, 0] - x_min
            pts[:, 1] = pts[:, 1] - y_min
            plothand(img_crop, pts, bones)
            if savefig:
                outpath_img = './sample/%.5d_l.jpg' % sample_id
                cv2.imwrite(outpath_img, img_crop)
            else:
                cv2.imshow('sample', img_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        if np.sum(right_hand_uv[:, 2]) >= 12:
            pts = right_hand_uv.astype('float')
            x_min = min(pts[:, 0])
            x_max = max(pts[:, 0])
            y_min = min(pts[:, 1])
            y_max = max(pts[:, 1])
            x_min = x_min - (x_max - x_min) / 2
            x_max = x_max + (x_max - x_min) / 2
            y_min = y_min - (y_max - y_min) / 2
            y_max = y_max + (y_max - y_min) / 2
            x_min, x_max = max(0, x_min), min(image.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(image.shape[0], y_max)
            img_crop = image[y_min:y_max, x_min:x_max]
            pts[:, 0] = pts[:, 0] - x_min
            pts[:, 1] = pts[:, 1] - y_min
            plothand(img_crop, pts, bones)
            if savefig:
                outpath_img = './sample/%.5d_r.jpg' % sample_id
                cv2.imwrite(outpath_img, img_crop)
            else:
                cv2.imshow('sample', img_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        itr += 1


def main():
    viz(savefig=False)


if __name__ == '__main__':
    main()
