import os

import cv2
import numpy as np

from utils import plothand

depth_in = np.array([[475.62, 0, 311.125], [0, 475.62, 245.965], [0, 0, 1]])
eye = np.eye(3)
ex = np.array([24.7, -0.0471401, 3.72045]).reshape((3, 1))
color_ex = np.hstack((eye, ex))
color_in = np.array([[617.173, 0, 315.453], [0, 617.173, 242.259], [0, 0, 1]])


def read_kp_color(path):
    """
    read the key point annotation from path and return in uvd form
    Args:
        :param path: path to the single annotation file

    Returns:
        :return: key points annotation in uvd form
    """
    with open(path) as file:
        file_content = file.read().replace('\n', '')
        list = file_content.split(',')
        list2 = [[float(list[i]), float(list[i + 1]), float(list[i + 2])] for i in range(0, len(list), 3)]
    kp_3d = np.array(list2).T
    ones = np.ones((1, 21))
    ap = np.vstack((kp_3d, ones))
    kp_cam = np.matmul(color_ex, ap)
    kp_color = np.matmul(color_in, kp_cam)
    kp_color[:2, :] = kp_color[:2, :] / kp_color[2:, :]
    return kp_color.T


def viz(inpath, outpath, save_fig=False):
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
            kp_path = dir + frame[0:8] + '_joint_pos.txt'
            kp = read_kp_color(kp_path)
            img = cv2.imread(imgpath)
            plothand(img, kp)
            if save_fig:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(outpath, frame[0:8] + '.jpg'), img)
            else:
                cv2.imshow(frame, img)


def main():
    inpath = 'male_object/seq01/cam01/'
    outpath = 'out_male_object/cam01/'
    viz(inpath, outpath)


if __name__ == '__main__':
    main()
