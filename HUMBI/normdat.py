import os
import numpy as np
import cv2
import json
from utils import generate_json_3d
from HUMBI.utils import *


def extract_sample(camera_list, inpath, outpath, is_left=True):
    """
    viz_sample is a function that visualize key points annotation of a single hand from this dataset

    Args:
        :param camera_list: camera calibration matrix
        :param inpath: input path of a particular frame, such as 'HUMBI/subject_1/hand/00000001'
        :param outpath: output path of formatted dataset
        :param is_left: whether the hand is left hand.
    Returns:
        :return: None
    """
    if is_left:
        indir = inpath + '/image_cropped/left/'
        kp_path = inpath + '/reconstruction/keypoints_l.txt'
        outpath = outpath + '/left/'
        is_left = 1
    else:
        indir = inpath + '/image_cropped/right/'
        kp_path = inpath + '/reconstruction/keypoints_r.txt'
        outpath = outpath + '/right/'
        is_left = 0
    kp_3d = read_kp_3d(kp_path)
    with open(indir + 'list.txt') as file:
        file_content = file.read()
        crop_info = file_content.split('\n')
    for info in crop_info:
        if info != '':
            id = int(info.split(' ')[0])
            img_name = 'image%07d.png' % id
            bbox = [float(num) for num in info.split(' ')[1:5]]

            scale_x = (bbox[1] - bbox[0] + 1) / 250
            scale_y = (bbox[3] - bbox[2] + 1) / 250
            # find corresponding camera matrix
            for qq in (item for item in camera_list if item['id'] == id):
                camera_param = qq
            M = camera_param['project']
            C = camera_param['C']
            R = camera_param['R']
            K = camera_param['intrinsic']
            T = - np.matmul(R, C)
            ex = np.hstack((R, T))  # 3*4
            M2 = np.matmul(K, ex)

            kp_2d = project2d(kp_3d, M2)  # 21*2
            xyz = project_camera(kp_3d, ex)

            kp_2d[:, 0] = (kp_2d[:, 0] - bbox[0]) / scale_x
            kp_2d[:, 1] = (kp_2d[:, 1] - bbox[2]) / scale_y

            # construct json info
            x_max = max(kp_2d[:, 0])
            x_min = min(kp_2d[:, 0])
            y_max = max(kp_2d[:, 1])
            y_min = min(kp_2d[:, 1])
            hand_bbox = [x_min, x_max, y_min, y_max]
            dict_kp = generate_json_3d(kp_2d, xyz, hand_bbox, is_left)

            if os.path.exists(indir + img_name):
                img = cv2.imread(indir + img_name)
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outpath_img = outpath + 'image%07d.jpg' % id
                outpath_json = outpath + 'image%07d.json' % id
                cv2.imwrite(outpath_img, img)
                with open(outpath_json, 'w') as outfile:
                    json.dump(dict_kp, outfile)


def main():
    dir = 'Hand_1_80_updat/'
    dir2 = 'Hand_81_140_updat/'
    dir3 = 'Hand_221_300_updat/'
    dir4 = 'Hand_301_380_updat/'
    dir5 = 'Hand_381_453_updat/'
    out = 'data/'
    for sub in os.listdir(dir):
        sub = dir + sub
        out1 = out + sub
        print(sub)
        if os.path.isdir(sub):
            hand_dir = sub + '/hand/'
            out2 = out1 + '/hand/'
            camera_list = read_param(hand_dir)
            for d in os.listdir(hand_dir):
                in_dir = hand_dir + d
                out_dir = out2 + d
                print(out_dir)
                if os.path.isdir(in_dir):
                    extract_sample(camera_list, in_dir, outpath=out_dir, is_left=True)
                    extract_sample(camera_list, in_dir, outpath=out_dir, is_left=False)


if __name__ == '__main__':
    main()
