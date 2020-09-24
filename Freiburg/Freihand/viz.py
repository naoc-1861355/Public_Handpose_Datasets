from __future__ import print_function, unicode_literals

import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from Freiburg.Freihand.freihand_utils.fh_utils import *

plt.rcParams['figure.figsize'] = (10, 10)


def show_training_samples(base_path, version, num2show=None, savefig=False):
    """
    show_training_samples is a function that visualize samples from one subset of training set of this dataset

    Args:
        :param base_path: path to the dataset
        :param version: a version of dataset, including ('gs', 'hom', 'sample', 'auto')
        :param num2show: number of samples to convert to ezxr format. If num2show = -1, then convert the whole dataset
        :param savefig: whether to save the visualized image. default to False

    Returns:
        :return: None
    """
    if num2show == -1:
        num2show = db_size('training')  # show all

    # load annotations
    db_data_anno = load_db_annotation(base_path, 'training')

    # iterate over all samples
    for idx in range(db_size('training')):
        if idx >= num2show:
            break

        # load image and mask
        img = read_img(idx, base_path, 'training', version)
        org = read_org(idx, base_path, version)
        msk = read_msk(idx, base_path)
        # annotation for this frame
        # in python 2, list return a list; in python 3, zip return iterable object
        # original version in python 2, show convert to list in python 3
        db_data_anno = list(db_data_anno)
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = project_points(xyz, K)

        # render an image of the shape
        msk_rendered = None

        # show
        plt.clf()
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, wspace=0.05)

        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax3 = fig.add_subplot(spec[0, 2])
        ax2.imshow(img)
        ax1.imshow(org)
        ax3.imshow(msk if msk_rendered is None else msk_rendered)
        plot_hand(ax3, uv, order='uv')
        plot_hand(ax2, uv, order='uv')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        if savefig:
            outpath = os.path.join('./sample', '%08d.jpg' % sample_version.map_id(idx, version))
            plt.savefig(outpath, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def show_eval_samples(base_path, num2show=None, savefig=False):
    """
    show_eval_samples is a function that visualize samples from eval set of this dataset

    Args:
        :param base_path: path to the dataset
        :param num2show: number of samples to convert to ezxr format. If num2show = -1, then convert the whole dataset
        :param savefig: whether to save the visualized image. default to False

    Returns:
        :return: None
    """
    if num2show == -1:
        num2show = db_size('evaluation')  # show all

    for idx in range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        if savefig:
            outpath = os.path.join('./sample', '%08d.jpg' % idx)
            plt.savefig(outpath, bbox_inches='tight')
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.imshow(img)
            ax1.axis('off')
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--show_eval', action='store_true',
                        help='Shows samples from the evaluation split if flag is set, shows training split otherwise.')
    parser.add_argument('--mano', action='store_true',
                        help='Enables rendering of the hand if mano is available. See README for details.')
    parser.add_argument('--num2show', type=int, default=-1,
                        help='Number of samples to show. ''-1'' defaults to show all.')
    parser.add_argument('--sample_version', type=str, default=sample_version.gs,
                        help='Which sample version to use when showing the training set.'
                             ' Valid choices are %s' % sample_version.valid_options())
    args = parser.parse_args()

    # check inputs
    msg = 'Invalid choice: ''%s''. Must be in %s' % (args.sample_version, sample_version.valid_options())
    assert args.sample_version in sample_version.valid_options(), msg

    if args.show_eval:
        """ Show some evaluation samples. """
        show_eval_samples(args.base_path,
                          num2show=args.num2show)

    else:
        """ Show some training samples. """
        show_training_samples(
            args.base_path,
            args.sample_version,
            num2show=args.num2show,
            savefig=False
        )


if __name__ == '__main__':
    main()
