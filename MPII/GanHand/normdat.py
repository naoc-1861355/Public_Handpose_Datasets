import json
import os

import cv2
import numpy as np

from utils import generate_json_3d


def read_kp(path):
    """
    read the key point annotation in 2d
    Args:
        :param path: path to the single annotation file

    Returns:
        :return: key points annotation in 2d
    """
    with open(path) as file:
        file_content = file.read().replace('\n', '')
        list = file_content.split(',')
        list2 = [[float(list[i]), float(list[i + 1])] for i in range(0, len(list), 2)]
        kp = np.array(list2)
    return kp


def read_kp_3d(path):
    """
    read the key point annotation in 3d
    Args:
        :param path: path to the single annotation file

    Returns:
        :return: key points annotation in 3d
    """
    with open(path) as file:
        file_content = file.read().replace('\n', '')
        list = file_content.split(',')
        list2 = [[float(list[i]), float(list[i + 1]), float(list[i + 2])] for i in range(0, len(list), 3)]
        kp = np.array(list2)
    return kp


def normdat(inpath, outpath):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param inpath: path to this dataset
        :param outpath: output path of the formatted files
    Returns:
        :return: None
    """
    for sub_dir in os.listdir(inpath):
        dir = os.path.join(inpath, sub_dir)
        out_dir = os.path.join(outpath, sub_dir)
        frame_list = os.listdir(dir)
        frame_list = [frame for frame in frame_list if frame.endswith('.png')]
        for frame in frame_list:
            imgpath = os.path.join(dir, frame)
            kp_path = dir + frame[0:4] + '_joint2D.txt'
            kp_path_3d = dir + frame[0:4] + '_joint_pos.txt'
            kp_2d = read_kp(kp_path)
            kp_3d = read_kp_3d(kp_path_3d)
            img = cv2.imread(imgpath)

            x_max = max(kp_2d[:, 0])
            x_min = min(kp_2d[:, 0])
            y_max = max(kp_2d[:, 1])
            y_min = min(kp_2d[:, 1])
            hand_bbox = [x_min, x_max, y_min, y_max]
            dict_kp = generate_json_3d(kp_2d, kp_3d, hand_bbox, -1)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            outpath_img = os.path.join(out_dir, frame[0:4] + '.jpg')
            outpath_json = os.path.join(out_dir, frame[0:4] + '.json')
            cv2.imwrite(outpath_img, img)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp, outfile)


def main():
    outpath = 'norm_data/withObject/'
    inpath = 'data/withObject/'
    normdat(inpath, outpath)


if __name__ == '__main__':
    main()
