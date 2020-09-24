import os

import cv2
import numpy as np

from utils import plothand


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


def viz(inpath, outpath='sample/', save_fig=False):
    """
    viz_sample is a function that visualize key points annotation of a single hand from this dataset

    Args:
        :param inpath: path to this dataset
        :param outpath: output path of the visualized image, if save_fig is True
        :param save_fig: whether to save the visualized image. Default to False

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
            kp = read_kp(kp_path)
            img = cv2.imread(imgpath)
            plothand(img, kp)
            if save_fig:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(outpath, frame[0:4]+'.jpg'), img)
            else:
                cv2.imshow(frame, img)


def main():
    inpath = 'data/withObject/'
    viz(inpath, save_fig=False)


if __name__ == '__main__':
    main()
