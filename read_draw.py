import json
import logging
import os

import cv2
import numpy as np


def read_json(json_path, task_id):
    """
    read_json is a function that read ezxr format json file
    Args:
        :param json_path: json filepath that contains keypoints annotation in ezxr format
        :param task_id: task_id of this reading. Useless when there's no parallel reading
    Returns:
        :return: 2d joints annotation (21, 2)
    """
    try:
        if not os.path.exists(json_path):
            return None, None
        json_file = open(json_path, 'r')
        kp_labels = json.load(json_file)
        json_file.close()
    except Exception:
        logging.error('Task-%d  failed to load label: %s', (task_id, json_path))
        return None, None

    kp_positions = np.zeros((21, 2), dtype='float32')
    kp_depth = np.zeros((21, 1), dtype='float32') - 1
    # read ezxr_render json label
    ps = [kp_labels['f52'][0]['x'] + 0.5, kp_labels['f52'][0]['y'] + 0.5]
    kp_positions[0] = np.asarray(ps)
    if 'f53' in kp_labels:
        kp_depth[0] = np.asarray([kp_labels['f53'][0]['z']])
    else:
        kp_depth[0] = np.asarray([-1.0])
    jid = 1
    for fid in range(5):
        ps = kp_labels['f%d2' % fid]
        if 'f53' in kp_labels:
            ps_3d = kp_labels['f%d3' % fid]
        for i in range(4):
            p = [ps[i]['x'] + .5, ps[i]['y'] + 0.5]
            kp_positions[jid] = np.asarray(p)
            if 'f53' in kp_labels:
                kp_depth[jid] = np.asarray([ps_3d[i]['z']])
            else:
                kp_depth[0] = np.asarray([-1.0])
            jid += 1
    return kp_positions


def draw(hand_joints, img_path):
    """
    draw is a function that draws joint annotations on given (corresponding) image
    The given annotation should correspond to the given image, or the behavior of this function is not guaranteed
    Args:
        :param hand_joints: numpy or array joints annotation (21, 2)
        :param img_path: image path of the given image
    Returns:
        :return: None
    """
    img_frame = cv2.imread(img_path)
    for fid in range(5):
        for jid in range(4):
            if jid == 0:
                pt1 = (int(hand_joints[0, 0]), int(hand_joints[0, 1]))
            else:
                pt1 = (int(hand_joints[fid * 4 + jid, 0]), int(hand_joints[fid * 4 + jid, 1]))
            if fid != 2 and fid != 3 or jid != 0:
                pt2 = (int(hand_joints[fid * 4 + jid + 1, 0]), int(hand_joints[fid * 4 + jid + 1, 1]))
                cv2.line(img_frame, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    for jid in range(1, 4):
        pt1 = (int(hand_joints[1 + jid * 4, 0]), int(hand_joints[1 + jid * 4, 1]))
        pt2 = (int(hand_joints[1 + (jid + 1) * 4, 0]), int(hand_joints[1 + (jid + 1) * 4, 1]))
        cv2.line(img_frame, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
    for jid in range(0, 21):
        joint = hand_joints[jid, :]
        jpos = (int(joint[0]), int(joint[1]))
        cv2.circle(img_frame, jpos, 2, (0, 255, 0), -1)
    cv2.imshow('1', img_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    filepath = 'data/Hand_381_453_updat/subject_391/hand/00000025/left/'
    joints = read_json(filepath + 'image0000075.json', 1)
    draw(joints, filepath + 'image0000075.jpg')


if __name__ == '__main__':
    main()
