import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('--config', default='mask-rcnn_r50_fpn_1x_coco.py')
    parser.add_argument(
        '--prediction_path',default='maskrcnn.pkl')
    parser.add_argument(
        '--save_dir',
        default='./')
    parser.add_argument(
        '--show', action='store_true')
    parser.add_argument(
        '--color-theme',
        default='plasma')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3)
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5)
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction)
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):

    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):

    true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:
            confusion_matrix[gt_label, -1] += 1


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='plasma'):

    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    ax.grid(True, which='minor', linestyle='-')

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    int(confusion_matrix[
                        i,
                        j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)

    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    plot_confusion_matrix(
        confusion_matrix,
        dataset.metainfo['classes'] + ('background', ),
        save_dir=args.save_dir,
        show=args.show,
        color_theme=args.color_theme)


    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP

    recall = TP / (TP + FP)
    precision = TP / (TP + FN)
    f1 =  2 * (precision * recall) / (precision + recall)
    average_precision = sum(precision) / 7
    average_recall = sum(recall) / 7

    average_f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall)
    accuracy = np.sum(TP) / np.sum(confusion_matrix)

    print('/n')
    print(dataset.metainfo['classes'] + ('background', ))

    print(" precision ", precision)
    print("recall ", recall)
    print("f1 ", f1)

    print("average P ", average_precision)
    print("average R", average_recall)
    print("average F1", average_f1)
    print("accuracy ", accuracy)


if __name__ == '__main__':
    main()
