from InterHand.utils import *
import os
import json
import numpy as np
import cv2
from utils import hand_angle


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
    """
    Given two bounding boxes, compute the iou (intersection of union) of them

    :param bbox1: bounding box 1
    :param bbox2: bounding box 2
    :return: the iou of the two bounding boxes
    """
    colInt = min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0])
    rowInt = min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2])
    intersection = colInt * rowInt
    area1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2])
    area2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2])
    return intersection / (area1 + area2 - intersection)


def generate_json(kp_2d, kp_3d, bbox, hand_type, interacting):
    is_left = (1 if hand_type == 'left' else 0)
    interact = (1 if interacting else 0)

    f52 = {'x': kp_2d[20, 0], 'y': kp_2d[20, 1], 'd': -1}
    f02 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in kp_2d[3::-1, :]]
    f12 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in kp_2d[7:3:-1, :]]
    f22 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in kp_2d[11:7:-1, :]]
    f32 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in kp_2d[15:11:-1, :]]
    f42 = [{'x': pt[0], 'y': pt[1], 'd': -1} for pt in kp_2d[19:15:-1, :]]
    f53 = {'x': kp_3d[20, 0], 'y': kp_3d[20, 1], 'z': kp_3d[20, 2]}
    f03 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in kp_3d[3::-1, :]]
    f13 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in kp_3d[7:3:-1, :]]
    f23 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in kp_3d[11:7:-1, :]]
    f33 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in kp_3d[15:11:-1, :]]
    f43 = [{'x': pt[0], 'y': pt[1], 'z': pt[2]} for pt in kp_3d[19:15:-1, :]]
    dict_kp = {'palm_center': [-1, -1, -1], 'is_left': is_left, 'interacting': interact, 'hand_bbox': bbox,
               'f52': [f52 for _ in range(4)], 'f53': [f53 for _ in range(4)],
               'f03': f03, 'f13': f13, 'f23': f23, 'f33': f33, 'f43': f43,
               'f02': f02, 'f12': f12, 'f22': f22, 'f32': f32, 'f42': f42
               }
    return dict_kp


def load_save_info(mode, annot_subset, banlist, outdir):
    """
    load annotation information from the datatset and save to ezxr format
    Args:
        :param outdir: output dir
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
    print('outputpath should be ' + os.path.join(outdir, mode))
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
            joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(joint_num * 2)
            if ann['hand_type'] != 'interacting':
                hand = joint_type[ann['hand_type']]
                kp_2d = joints_img[hand]
                kp_cam = joints_cam[hand]
                angle = hand_angle(kp_cam)
                img_path = os.path.join(img_dir, mode, img['file_name'])
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    if np.sum(joint_valid[hand]) > 18 and angle < 50 and np.sum(image) > 0:
                        out_path = os.path.join(outdir, mode, img['file_name'])
                        os.makedirs(out_path[:-14], exist_ok=True)
                        dict_kp = generate_json(kp_2d, kp_cam, bbox, ann['hand_type'], interacting=False)
                        cv2.imwrite(out_path[:-3] + 'jpg', image)
                        with open(out_path[:-3] + 'json', 'w') as outfile:
                            json.dump(dict_kp, outfile)
            else:
                if np.sum(joint_valid) > 36:
                    joints_img_left = joints_img[joint_type['left']]
                    joints_img_right = joints_img[joint_type['right']]
                    joints_cam_left = joints_cam[joint_type['left']]
                    joints_cam_right = joints_cam[joint_type['right']]
                    angle_left = hand_angle(joints_cam_left)
                    angle_right = hand_angle(joints_cam_right)
                    bbox_left = [max(min(joints_img_left[:, 0]) - 10, 0.0), min(max(joints_img_left[:, 0]) + 10, 334.0),
                                 max(min(joints_img_left[:, 1]) - 10, 0.0), min(max(joints_img_left[:, 1]) + 10, 512.0)]
                    bbox_right = [max(min(joints_img_right[:, 0]) - 10, 0.0),
                                  min(max(joints_img_right[:, 0]) + 10, 334.0),
                                  max(min(joints_img_right[:, 1]) - 10, 0.0),
                                  min(max(joints_img_right[:, 1]) + 10, 512.0)]
                    iou_perc = iou(bbox_left, bbox_right)
                    if iou_perc < 0.1:
                        img_path = os.path.join(img_dir, mode, img['file_name'])
                        out_path = os.path.join(outdir, mode, img['file_name'])
                        if os.path.exists(img_path):
                            image = cv2.imread(img_path)
                            if angle_left < 50 and np.sum(image) > 1000:
                                os.makedirs(out_path[:-14], exist_ok=True)
                                dict_kp = generate_json(joints_img_left, joints_cam_left, bbox_left, 'left',
                                                        interacting=True)
                                cv2.imwrite(out_path[:-4] + '_left.jpg', image)
                                with open(out_path[:-4] + '_left.json', 'w') as outfile:
                                    json.dump(dict_kp, outfile)
                            if angle_right < 50 and np.sum(image) > 1000:
                                os.makedirs(out_path[:-14], exist_ok=True)
                                dict_kp = generate_json(joints_img_right, joints_cam_right, bbox_right, 'right',
                                                        interacting=True)
                                cv2.imwrite(out_path[:-4] + '_right.jpg', image)
                                with open(out_path[:-4] + '_right.json', 'w') as outfile:
                                    json.dump(dict_kp, outfile)
        print(idx)


def main():
    mode = 'val'
    subset = 'machine_annot'
    outdir = '../InterHand2.6M_ezxr_2'
    # the following list contains invalid camera(contains occlusion)
    test_banlist = ['cam400067', 'cam400006', 'cam400008', 'cam410218', 'cam410236', 'cam410063',
                    'cam400035', 'cam410001', 'cam410028', 'cam410210', 'cam400015', 'cam400029', 'cam400049']
    train_banlist = ['cam400006', 'cam400008', 'cam400015', 'cam400029', 'cam400035', 'cam400049', 'cam410019',
                     'cam410028', 'cam410063', 'cam410210', 'cam410218', 'cam410236']
    load_save_info(mode, subset, test_banlist, outdir)


if __name__ == '__main__':
    main()
