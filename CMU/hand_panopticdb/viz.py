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
    outpath = './sample/'

    # Input data paths
    folder_path = 'D:\Hand-data\CMU\hand143_panopticdb/'  # Put your local path here
    json_path = folder_path + 'hands_v143_14817.json'

    with open(json_path, 'r') as fid:
        dat_all = json.load(fid)
        dat_all = dat_all['root']

    # dat = dat_all[0]  # Choose one element as an example

    for dat in dat_all[0:3]:
        pts = np.array(dat['joint_self'], dtype=float)
        invalid = pts[:, 2] != 1

        imgpath = dat['img_paths']
        imgpath = folder_path + imgpath
        print(imgpath)
        img = cv2.imread(imgpath)

        # find bounding point for each img (hand)
        x_min = min(pts[:, 0])
        x_max = max(pts[:, 0])
        y_min = min(pts[:, 1])
        y_max = max(pts[:, 1])
        x_min = x_min - (x_max - x_min) / 2
        x_max = x_max + (x_max - x_min) / 2
        y_min = y_min - (y_max - y_min) / 2
        y_max = y_max + (y_max - y_min) / 2
        x_min, x_max = max(0, x_min), min(img.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(img.shape[0], y_max)
        hand_bbox = [x_min, x_max, y_min, y_max]

        # Plot annotations
        img_crop = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        pts[:, 0] = pts[:, 0] - x_min
        pts[:, 1] = pts[:, 1] - y_min

        plothand(img_crop, pts)
        if savefile:
            if not os.path.isdir(outpath):
                os.makedirs(outpath)
            cv2.imwrite(outpath + dat['img_paths'][5:-4] + '.jpg', img_crop)
        else:
            cv2.imshow(dat['img_paths'][5:-4], img_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    viz(savefile=True)


if __name__ == '__main__':
    main()
