import json
import os

import cv2
import numpy as np

from utils import generate_json_3d

depth_in = np.array([[475.62, 0, 311.125], [0, 475.62, 245.965], [0, 0, 1]])
eye = np.eye(3)
ex = np.array([24.7, -0.0471401, 3.72045]).reshape((3, 1))
color_ex = np.hstack((eye, ex))
color_in = np.array([[617.173, 0, 315.453], [0, 617.173, 242.259], [0, 0, 1]])


def read_kp(path):
    """
    read the key point annotation from path and return in uvd form
    Args:
        :param path: path to the single annotation file

    Returns:
        :return: key points annotation in uvd form
        :return: key points annotation in 3d form
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
    return kp_color.T, kp_3d.T


def normdat(inpath, outpath):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param inpath: path to this dataset
        :param outpath: output path of the formatted files
    Returns:
        :return: None
    """
    sequences = os.listdir(inpath)
    # exclude = ['seq01', 'seq02', 'seq03', 'seq04']
    # seqs = [i for i in sequences if i not in exclude]

    for seq in sequences:
        inpath_dir1 = os.path.join(inpath, seq)
        outpath1 = os.path.join(outpath, seq)
        for cam in os.listdir(inpath_dir1):
            inpath_dir2 = os.path.join(inpath_dir1, cam)
            outpath2 = os.path.join(outpath1, cam)
            for i in os.listdir(inpath_dir2):
                inpath_dir3 = os.path.join(inpath_dir2, i)
                outpath3 = os.path.join(outpath2, i)
                if not os.path.exists(outpath3):
                    os.makedirs(outpath3)
                imgs_name = os.listdir(inpath_dir3)
                txt_names = [i for i in imgs_name if '.txt' in i]
                for txt_name in txt_names:
                    j = int(txt_name[0:8])
                    kp_path = inpath_dir3 + '/%08d_joint_pos.txt' % j
                    imgpath = inpath_dir3 + '/%08d_color.png' % j
                    kp_2d, kp_3d = read_kp(kp_path)

                    x_max = max(kp_2d[:, 0])
                    x_min = min(kp_2d[:, 0])
                    y_max = max(kp_2d[:, 1])
                    y_min = min(kp_2d[:, 1])
                    hand_bbox = [x_min, x_max, y_min, y_max]
                    dict_kp = generate_json_3d(kp_2d, kp_3d, hand_bbox, -1)

                    with open(outpath3 + '/%08d.json' % j, 'w') as outfile:
                        json.dump(dict_kp, outfile)
                    img = cv2.imread(imgpath)
                    cv2.imwrite(outpath3 + '/%08d.jpg' % j, img)


def main():
    inpath = 'female_object/'
    outpath = 'data/female_object/'
    normdat(inpath, outpath)


if __name__ == '__main__':
    main()
