from __future__ import print_function, unicode_literals

import json
import os
import pickle
from shutil import copyfile

import numpy as np

# chose between training and evaluation set
set = 'training'
# set = 'evaluation'


def normdat(inpath, outpath):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param inpath: path to the dataset
        :param outpath: path of the output files

    Returns:
        :return: None
    """
    # load annotations of this set
    with open(os.path.join(inpath, set, 'anno_%s.pickle' % set), 'rb') as fi:
        anno_all = pickle.load(fi)

    # iterate samples of the set
    itr = 0
    for sample_id, anno in anno_all.items():
        # load data
        impath_img = os.path.join(inpath, set, 'color', '%.5d.png' % sample_id)
        # get info from annotation dictionary
        kp_coord_uv = anno['uv_vis']  # u, v coordinates of 42 hand keypoints, pixel
        kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

        # Project world coordinates into the camera frame
        kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
        kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

        # split to left and right hand
        left_hand_uv = kp_coord_uv[0:21]
        right_hand_uv = kp_coord_uv[21:42]
        left_hand_xyz = kp_coord_uv_proj[0:21]
        right_hand_xyz = kp_coord_uv_proj[21:42]

        # poccess left hand
        # hand must contain at least 12 kpt to be a valid hand
        if (np.sum(left_hand_uv[:, 2]) >= 12):
            pts = left_hand_uv.astype('float')
            xyz = left_hand_xyz.astype('float')
            x_min = min(pts[:, 0])
            x_max = max(pts[:, 0])
            y_min = min(pts[:, 1])
            y_max = max(pts[:, 1])
            hand_bbox = [x_min, x_max, y_min, y_max]
            f52 = {'x': pts[0, 0], 'y': pts[0, 1], 'd': -1}
            f02 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[4:0:-1, :]]
            f12 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[8:4:-1, :]]
            f22 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[12:8:-1, :]]
            f32 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[16:12:-1, :]]
            f42 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[20:16:-1, :]]

            f53 = {'x': xyz[0, 0], 'y': xyz[0, 1], 'z': xyz[0, 2]}
            f03 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[4:0:-1, :]]
            f13 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[8:4:-1, :]]
            f23 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[12:8:-1, :]]
            f33 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[16:12:-1, :]]
            f43 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[20:16:-1, :]]

            dict_kp_l = {'palm_center': [-1, -1, -1], 'is_left': 1, 'hand_bbox': hand_bbox,
                         'f52': [f52 for _ in range(4)], 'f53': [f53 for _ in range(4)],
                         'f03': f03, 'f13': f13, 'f23': f23, 'f33': f33, 'f43': f43,
                         'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42}
            outpath_json = os.path.join(outpath, '%.5d_l.json' % sample_id)
            outpath_img = os.path.join(outpath, '%.5d_l.jpg' % sample_id)
            copyfile(impath_img, outpath_img)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp_l, outfile)
        # poccess right hand
        # hand must contain at least 12 kpt to be a valid hand
        if (np.sum(right_hand_uv[:, 2]) >= 12):
            pts = right_hand_uv.astype('float')
            xyz = right_hand_xyz.astype('float')
            x_min = min(pts[:, 0])
            x_max = max(pts[:, 0])
            y_min = min(pts[:, 1])
            y_max = max(pts[:, 1])
            hand_bbox = [x_min, x_max, y_min, y_max]
            f52 = {'x': pts[0, 0], 'y': pts[0, 1], 'd': -1}
            f02 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[4:0:-1, :]]
            f12 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[8:4:-1, :]]
            f22 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[12:8:-1, :]]
            f32 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[16:12:-1, :]]
            f42 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in pts[20:16:-1, :]]

            f53 = {'x': xyz[0, 0], 'y': xyz[0, 1], 'z': xyz[0, 2]}
            f03 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[4:0:-1, :]]
            f13 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[8:4:-1, :]]
            f23 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[12:8:-1, :]]
            f33 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[16:12:-1, :]]
            f43 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in xyz[20:16:-1, :]]

            dict_kp_l = {'palm_center': [-1, -1, -1], 'is_left': 0, 'hand_bbox': hand_bbox,
                         'f52': [f52 for _ in range(4)], 'f53': [f53 for _ in range(4)],
                         'f03': f03, 'f13': f13, 'f23': f23, 'f33': f33, 'f43': f43,
                         'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42}
            outpath_json = os.path.join(outpath, '%.5d_r.json' % sample_id)
            outpath_img = os.path.join(outpath, '%.5d_r.jpg' % sample_id)
            copyfile(impath_img, outpath_img)
            with open(outpath_json, 'w') as outfile:
                json.dump(dict_kp_l, outfile)
