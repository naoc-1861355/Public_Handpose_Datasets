from shutil import copyfile

from Freiburg.Freihand.freihand_utils.fh_utils import *
from utils import generate_json_3d


def normdat(base_path, outpath, version, num2show=None):
    """
    normdat is a function that convert this dataset to standard ezxr format output

    Args:
        :param base_path: path to the dataset
        :param outpath: path of the output files
        :param version: a version of dataset, including ('gs', 'hom', 'sample', 'auto')
        :param num2show: number of samples to convert to ezxr format. If num2show = -1, then convert the whole dataset

    Returns:
        :return: None
    """
    if num2show == -1:
        num2show = db_size('training')  # show all

    # load annotations
    db_data_anno = load_db_annotation(base_path, 'training')
    # in python 2, list return a list; in python 3, zip return iterable object
    # original version in python 2, show convert to list in python 3
    db_data_anno = list(db_data_anno)

    # iterate over all samples
    for idx in range(db_size('training')):
        if idx >= num2show:
            break

        # annotation for this frame
        K, xyz = db_data_anno[idx]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = project_points(xyz, K)
        pts = uv

        x_min = min(pts[:, 0])
        x_max = max(pts[:, 0])
        y_min = min(pts[:, 1])
        y_max = max(pts[:, 1])
        hand_bbox = [x_min, x_max, y_min, y_max]

        dict_kp = generate_json_3d(pts, xyz, hand_bbox, is_left=-1)
        # copy and dump .jpg and .json
        outpath_json = os.path.join(outpath, 'training', '%08d.json' % sample_version.map_id(idx, version))
        with open(outpath_json, 'w') as outfile:
            json.dump(dict_kp, outfile)

        im_path = img_path(idx, base_path, 'training', version)
        outpath_img = os.path.join(outpath, 'training', '%08d.jpg' % sample_version.map_id(idx, version))
        copyfile(im_path, outpath_img)


def main():
    outpath = 'freihand_ezxr'
    normdat(
        'd:/Hand-data/Freiburg/FreiHAND',
        outpath,
        'auto',
        num2show=-1,
    )


if __name__ == '__main__':
    main()
