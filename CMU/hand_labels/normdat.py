import json
import os.path

import cv2
import numpy as np

from utils import generate_json_2d


def normdat(outpath):
    """

    Args:
        :param outpath : root output path of the formatted files

    Returns:
        :return: None
    """
    # Input data paths
    # paths = ['synth1/', 'synth2/', 'synth3/', 'synth4/']
    paths = ['manual_test/', 'manual_train/']
    inpath = paths[0]
    outpath = outpath + inpath

    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    files = sorted([f for f in os.listdir(inpath) if f.endswith('.json')])

    for f in files:
        with open(inpath + f, 'r') as fid:
            dat = json.load(fid)

        pts = np.array(dat['hand_pts'], dtype=float)

        # Left hands are marked, but otherwise follow the same point order
        is_left = dat['is_left']
        # find bounding point for each img (hand)
        x_min = min(pts[:, 0])
        x_max = max(pts[:, 0])
        y_min = min(pts[:, 1])
        y_max = max(pts[:, 1])
        hand_bbox = [x_min, x_max, y_min, y_max]

        dict_kp = generate_json_2d(pts, hand_bbox, is_left)
        # copy and dump .jpg and .json
        img = cv2.imread(inpath + f[0:-5] + '.jpg')
        cv2.imwrite(outpath + f[0:-5] + '.jpg', img)
        with open(outpath + f[0:-5] + '.json', 'w') as outfile:
            json.dump(dict_kp, outfile)


def main():
    outpath = './data/'
    normdat(outpath)


if __name__ == '__main__':
    main()
