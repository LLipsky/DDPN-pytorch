import torch
import numpy as np


def bbox_overlaps_batch(anchors, gt_boxes):
    """
        anchors: (N, 4) ndarray of float
        gt_boxes: (b, K, 5) ndarray of float,b means batchsize

        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)

    if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()  # 因为第三维只要前4个

        gt_boxes_x = (gt_boxes[:, :, 2]-gt_boxes[:, :, 0]+1)
        gt_boxes_y = (gt_boxes[:, :, 3]-gt_boxes[:, :, 1]+1)
        gt_boxes_area = (gt_boxes_x*gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2]-anchors[:, :, 0]+1)
        anchors_boxes_y = (anchors[:, :, 3]-anchors[:, :, 1]+1)
        anchors_area = (anchors_boxes_x*anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)  # predicted boxes
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)  # ground-truth boxes

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2])-torch.max(boxes[:, :, :, 0]-boxes[:, :, :, 0])+1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3])-torch.max(boxes[:, :, :, 1]-boxes[:, :, :, 1])+1)
        ih[ih < 0] = 0

        ua = anchors_area+gt_boxes_area-iw*ih  # A并B-A交B
        overlaps = iw*ih/ua  # A交B/（A并B-A交B）#为什么分母是这个？？？？？/,求重叠率

        # mask the overlap here#这里的mask具体是什么含义？？？？
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)  # function masked_fill_(mask,value)函数，Fills elements of self tensor with value where mask is one.
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)  # 假设a是一个2*3矩阵，shape(a),相当于返回矩阵a每维的大小（2，3）；size(a)是返回矩阵中的数据个数（6）
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()

        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2]-gt_boxes[:, :, 0]+1)
        gt_boxes_y = (gt_boxes[:, :, 3]-gt_boxes[:, :, 1]+1)
        gt_boxes_area = (gt_boxes_x*gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2]-anchors[:, :, 0]+1)
        anchors_boxes_y = (anchors[:, :, 3]-anchors[:, :, 1]+1)
        anchors_area = (anchors_boxes_x*anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)  # predicted boxes
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)  # ground-truth boxes

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) - torch.max(
            boxes[:, :, :, 0] - boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) - torch.max(
            boxes[:, :, :, 1] - boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0

        ua = anchors_area + gt_boxes_area - iw * ih  # A并B-A交B

        overlaps = iw * ih / ua  # A交B/（A并B-A交B）#为什么分母是这个？？？？？/

        # mask the overlap here#这里的mask具体是什么含义？？？？
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)  # 函数masked_fill_(mask,value)函数，Fills elements of self tensor with value where mask is one.
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps

def bbox_transform(ex_rois, gt_rois):  # ex_rois(means what????), gt_rois means groundtruth
    ex_widths = ex_rois[:, 2]-ex_rois[:, 0]+1.0
    ex_heights = ex_rois[:, 3]-ex_rois[:, 1]+1.0
    ex_ctr_x = ex_rois[:, 0]+0.5*ex_widths
    ex_ctr_y = ex_rois[:, 1]+0.5*ex_heights

    gt_widths = gt_rois[:, 2]-gt_rois[:, 0]+1.0
    gt_heights = gt_rois[:, 3]-gt_rois[:, 1]+1.0
    gt_ctr_x = gt_rois[:, 0]+0.5*gt_widths
    gt_ctr_y = gt_rois[:, 1]+0.5*gt_heights

    targets_dx = (gt_ctr_x-ex_ctr_x)/ex_widths
    targets_dy = (gt_ctr_y-ex_ctr_y)/ex_heights

    targets_dw = np.log(gt_widths/ex_widths)
    targets_dh = np.log(gt_heights/ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets











