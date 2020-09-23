import json
import os.path

import cv2
import numpy as np

from utils import plothand

edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
         [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


def viz(savefile=False):
    """
     viz is a function that crop and visualize keypoint annotation on image

     Args:
         :param savefile : whether to save the visualized image. default to False

     Returns:
        :return None
    """
    # output data paths. Replace your own output path here
    outpath = './sample/'

    # Input data paths. Replace your own input path here
    # paths = ['synth1/', 'synth2/', 'synth3', 'synth4']
    paths = ['manual_test/', 'manual_train/']
    inpath = paths[0]

    files = sorted([f for f in os.listdir(inpath) if f.endswith('.json')])

    for f in files[-2:-1]:
        with open(inpath + f, 'r') as fid:
            dat = json.load(fid)

        # Each file contains 1 hand annotation, with 21 points in
        # 'hand_pts' of size 21x3, following this scheme:
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format
        # The 3rd column is 1 if valid:
        pts = np.array(dat['hand_pts'])
        invalid = pts[:, 2] != 1
        im = cv2.imread(inpath + f[0:-5] + '.jpg')

        # find bounding box for each img (hand)
        x_max = 0
        y_max = 0
        x_min = 10000
        y_min = 10000
        for p in range(pts.shape[0]):
            if pts[p, 2] != 0:
                if pts[p, 0] < x_min:
                    x_min = pts[p, 0]
                if pts[p, 0] > x_max:
                    x_max = pts[p, 0]
                if pts[p, 1] < y_min:
                    y_min = pts[p, 1]
                if pts[p, 1] > y_max:
                    y_max = pts[p, 1]
        crop_x_min = max(0, int(x_min - ((x_max - x_min) / 2)))
        crop_x_max = min(im.shape[1] - 1, int(x_max + ((x_max - x_min) / 2)))
        crop_y_min = max(0, int(y_min - ((y_max - y_min) / 2)))
        crop_y_max = min(im.shape[0] - 1, int(y_max + ((y_max - y_min) / 2)))

        im_crop = im[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        pts[:, 0] = pts[:, 0] - crop_x_min
        pts[:, 1] = pts[:, 1] - crop_y_min
        # Plot annotations
        plothand(im_crop, pts)
        if savefile:
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
            cv2.imwrite(outpath + f[0:-5] + '.jpg', im_crop)
        else:
            cv2.imshow(f[0:-5], im_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    viz(savefile=False)


if __name__ == '__main__':
    main()
