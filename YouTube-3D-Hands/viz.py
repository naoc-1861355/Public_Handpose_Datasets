import json
import numpy as np
import os
import time
import cv2
from utils import plothand

"""
This program write in python3.5
"""

kpId2vertices = {
    4: [744],  # ThumbT
    8: [320],  # IndexT
    12: [443],  # MiddleT
    16: [555],  # RingT
    20: [672]  # PinkT
}

unavailable_list = [
    '1VyRsTPK_Qk',
    'gxtiESaM93I',
    '6u-5KSA-SUQ',
    '79N7Tn2fDjM',
    'Su0mLlax_0s'
]


def get_keypoints_from_mesh_ch(mesh_vertices, keypoints_regressed):
    """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
    keypoints = [0.0 for _ in range(21)]  # init empty list

    # fill keypoints which are regressed
    mapping = {0: 0,  # Wrist
               1: 5, 2: 6, 3: 7,  # Index
               4: 9, 5: 10, 6: 11,  # Middle
               7: 17, 8: 18, 9: 19,  # Pinky
               10: 13, 11: 14, 12: 15,  # Ring
               13: 1, 14: 2, 15: 3}  # Thumb

    for manoId, myId in mapping.items():
        keypoints[myId] = keypoints_regressed[manoId, :]

    # get other keypoints from mesh
    for myId, meshId in kpId2vertices.items():
        keypoints[myId] = mesh_vertices[meshId, :]

    keypoints = np.vstack(keypoints)

    return keypoints


def load_dataset(fp_data='D:\Hand-data\YouTube-3D-Hands/data/youtube_train.json'):
    """Load the YouTube dataset.

    Args:
        fp_data: Filepath to the json file.

    Returns:
        Hand mesh dataset.
    """
    with open(fp_data, "r") as file:
        data = json.load(file)

    return data


def retrieve_sample(data, ann_index):
    """Retrieve an annotation-image pair from the dataset.

    Args:
        data: Hand mesh dataset.
        ann_index: Annotation index.

    Returns:
        A sample from the hand mesh dataset.
    """
    ann = data['annotations'][ann_index]
    images = data['images']
    img_idxs = [im['id'] for im in images]

    img = images[img_idxs.index(ann['image_id'])]
    return ann, img


def viz(data, ann_index, regressor, save_fig=False, db_root='D:\Hand-data\YouTube-3D-Hands/data/',
        outpath='./data_train/'):
    """
    viz_sample is a function that visualize key points annotation of a single hand from this dataset

    Args:
        :param data: dataset loaded from json annotation files
        :param ann_index: annotation index
        :param regressor: regressor of mesh points
        :param save_fig: whether to save the visualized image. default to False
        :param db_root: path to the database
        :param outpath: output path of the visualized image, if save_fig is True
    :return: None
    """
    import numpy as np
    from os.path import join

    ann, img = retrieve_sample(data, ann_index)
    video_id = img['name'].split('/')[1]
    if video_id not in unavailable_list:
        inpath_img = join(db_root, img['name'])
        if os.path.exists(inpath_img):
            img = cv2.imread(inpath_img)
            vertices = np.array(ann['vertices'])
            x_min = min(vertices[:, 0])
            x_max = max(vertices[:, 0])
            y_min = min(vertices[:, 1])
            y_max = max(vertices[:, 1])
            crop_x_min = max(0, int(x_min - ((x_max - x_min) / 2)))
            crop_x_max = min(img.shape[1] - 1, int(x_max + ((x_max - x_min) / 2)))
            crop_y_min = max(0, int(y_min - ((y_max - y_min) / 2)))
            crop_y_max = min(img.shape[0] - 1, int(y_max + ((y_max - y_min) / 2)))
            img_crop = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            # compute 2d keypoints info
            temp = vertices + np.array([0.0, 0.0, 0.3])
            Jtr_x = np.matmul(regressor, temp[:, 0])
            Jtr_y = np.matmul(regressor, temp[:, 1])
            Jtr_z = np.matmul(regressor, temp[:, 2])
            kp = np.vstack([Jtr_x, Jtr_y, Jtr_z]).T  # 16*3

            vertices[:, 0] = vertices[:, 0] - crop_x_min
            vertices[:, 1] = vertices[:, 1] - crop_y_min
            kp[:, 0] = kp[:, 0] - crop_x_min
            kp[:, 1] = kp[:, 1] - crop_y_min

            pts = get_keypoints_from_mesh_ch(vertices, kp)  # 21*3
            plothand(img_crop, pts)
            if save_fig:
                outpath_img = outpath + str(ann['id']) + '.jpg'
                cv2.imwrite(outpath_img, img_crop)
            else:
                cv2.imshow(ann['id'], img_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print('missing: ' + inpath_img)


def main():
    data = load_dataset()
    regressor = np.load('right_hand.npy')
    for i in range(10):
        viz(data, i, regressor)
    print("Data keys:", [k for k in data.keys()])
    print("Image keys:", [k for k in data['images'][0].keys()])
    print("Annotations keys:", [k for k in data['annotations'][0].keys()])

    print("The number of images:", len(data['images']))
    print("The number of annotations:", len(data['annotations']))


if __name__ == '__main__':
    main()
