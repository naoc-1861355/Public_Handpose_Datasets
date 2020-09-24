import json
import os

import cv2
import numpy as np

from InterHand.utils import *
from utils import hand_angle


def test_id_match(imgs, anns):
    """
    test whether the img_data match the ann_data
    :param imgs: list of img dicts
    :param anns: list of ann dicts
    :return: None
    """
    for idx, img in enumerate(imgs):
        imageid_from_img = img['id']
        imageid_from_ann = anns[idx]['image_id']
        if imageid_from_ann != imageid_from_img:
            print('fail' + str(idx))


def draw_hand(image, kp_2d, angle, bbox, joint_num=21):
    """
    plot single hand according to given info
    :param image: numpy image
    :param kp_2d: 2d keypoints position (21*2)
    :param angle: angle of the hand
    :param bbox: bounding box of the hand
    :param joint_num: number of joints, default 21
    :return: None
    """
    for i in range(joint_num):
        cv2.circle(image, (int(kp_2d[i, 0]), int(kp_2d[i, 1])), 2, (0, 0, 255), -1)
    cv2.putText(image, str(angle), (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.rectangle(image, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])),
                  (0, 0, 255), 1)
    cv2.imshow('1', image)
    cv2.moveWindow('1', 1000, 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iou(bbox1, bbox2):
    colInt = min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0])
    rowInt = min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2])
    intersection = colInt * rowInt
    area1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2])
    area2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2])
    return intersection / (area1 + area2 - intersection)


def viz(mode, annot_subset, banlist):
    """
    load annotation information from the datatset and visualize key points annotation of a single hand

    Args:
        :param banlist: camera list that shouldn't be included
        :param mode: info mode, including train, test, val
        :param annot_subset: annotation subset, including all, human_annot, machine_annot

    :return: None
    """
    print('start')
    img_dir = '../InterHand2.6M_5fps_batch0/images'
    annot_path = '../InterHand2.6M_5fps_batch0/annotations'
    joint_num = 21  # single hand
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}

    # load annotation
    print("Load annotation from  " + os.path.join(annot_path, annot_subset))
    with open(os.path.join(annot_path, annot_subset, 'InterHand2.6M_' + mode + '_data.json')) as f:
        data = json.load(f)
    with open(os.path.join(annot_path, annot_subset, 'InterHand2.6M_' + mode + '_camera.json')) as f:
        cameras = json.load(f)
    with open(os.path.join(annot_path, annot_subset, 'InterHand2.6M_' + mode + '_joint_3d.json')) as f:
        joints = json.load(f)

    imgs = data['images']
    anns = data['annotations']

    for idx, img in enumerate(imgs):
        if img['file_name'].split('/')[-2] not in banlist:
            ann = anns[idx]
            cam = img['camera']
            capture_id = img['capture']
            fram_id = img['frame_idx']
            bbox = ann['bbox']
            bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]

            joints_world = np.array(joints[str(capture_id)][str(fram_id)]['world_coord'])
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(
                cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)

            joints_cam = world2cam(joints_world.transpose((1, 0)), camrot, campos.reshape(3, 1)).transpose((1, 0))
            joints_img = cam2pixel(joints_cam, focal, princpt)[:, :2]
            print(ann)
            print(img)
            joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(joint_num * 2)
            if ann['hand_type'] != 'interacting':
                hand = joint_type[ann['hand_type']]
                kp_2d = joints_img[hand]
                kp_cam = joints_cam[hand]
                angle = hand_angle(kp_cam)

                img_path = os.path.join(img_dir, mode, img['file_name'])
                image = cv2.imread(img_path)
                print(np.sum(image))
                draw_hand(image, kp_2d, angle, bbox)
            else:
                joints_img_left = joints_img[joint_type['left']]
                joints_img_right = joints_img[joint_type['right']]
                bbox_left = [min(joints_img_left[:, 0]) - 10, max(joints_img_left[:, 0]) + 10,
                             min(joints_img_left[:, 1]) - 10, max(joints_img_left[:, 1]) + 10]
                bbox_right = [min(joints_img_right[:, 0]) - 10, max(joints_img_right[:, 0]) + 10,
                              min(joints_img_right[:, 1]) - 10, max(joints_img_right[:, 1]) + 10]
                iou_perc = iou(bbox_left, bbox_right)

                img_path = os.path.join(img_dir, mode, img['file_name'])
                image = cv2.imread(img_path)
                draw_hand(image, joints_img_right, iou_perc, bbox_right)
                image = cv2.imread(img_path)
                draw_hand(image, joints_img_left, iou_perc, bbox_left)


def main():
    mode = 'test'
    subset = 'all'
    # the following list contains invalid camera(contains occlusion)
    test_banlist = ['cam400067', 'cam400006', 'cam400008', 'cam410218', 'cam410236', 'cam410063',
                    'cam400035', 'cam410001', 'cam410028', 'cam410210', 'cam400015', 'cam400029', 'cam400049']
    train_banlist = ['cam400006', 'cam400008', 'cam400015', 'cam400029', 'cam400035', 'cam400049', 'cam410019',
                     'cam410028', 'cam410063', 'cam410210', 'cam410218', 'cam410236']
    viz(mode, subset, test_banlist)


if __name__ == '__main__':
    main()
