import json
import os.path

import cv2
import numpy as np

from utils import generate_json_2d

edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
         [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


def normdat(outpath):
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    # Input data paths
    folderPath = 'D:\Hand-data\CMU\hand143_panopticdb/'  # Put your local path here
    jsonPath = folderPath + 'hands_v143_14817.json'

    with open(jsonPath, 'r') as fid:
        dat_all = json.load(fid)
        dat_all = dat_all['root']

    # dat = dat_all[0]  # Choose one element as an example;
    for dat in dat_all[0:1]:
        pts = np.array(dat['joint_self'], dtype=float)
        img_path = dat['img_paths']

        # find bounding point for each img (hand)
        x_min = min(pts[:, 0])
        x_max = max(pts[:, 0])
        y_min = min(pts[:, 1])
        y_max = max(pts[:, 1])
        hand_bbox = [x_min, x_max, y_min, y_max]

        dict_kp = generate_json_2d(pts, hand_bbox, is_left=-1)
        print(dict_kp)
        # copy and dump .jpg and .json
        img = cv2.imread(img_path)
        cv2.imwrite(outpath + dat['img_paths'][5:-4] + '.jpg', img)
        with open(outpath + dat['img_paths'][5:-4] + '.json', 'w') as outfile:
            json.dump(dict_kp, outfile)


def main():
    outpath = './sample/'
    normdat(outpath)


if __name__ == '__main__':
    main()
