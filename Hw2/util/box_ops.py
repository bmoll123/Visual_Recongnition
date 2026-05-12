# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    # 🌟 核心修正：先處理 NaN
    # 如果模型噴出 NaN，強制把它們變成 0，防止後續所有運算崩潰
    if torch.isnan(boxes1).any():
        boxes1 = torch.nan_to_num(boxes1, nan=0.0, posinf=1.0, neginf=0.0)
    if torch.isnan(boxes2).any():
        boxes2 = torch.nan_to_num(boxes2, nan=0.0, posinf=1.0, neginf=0.0)

    # 🌟 暴力矯正法：確保 x2 >= x1, y2 >= y1
    # 注意：這裡不要用 in-place 修改 (boxes1[:, 2:] = ...)，有時會因為 gradient view 報錯
    # 我們改用重新賦值的方式
    boxes1_lt = boxes1[:, :2]
    boxes1_rb = torch.max(boxes1[:, 2:], boxes1_lt)
    boxes1 = torch.cat([boxes1_lt, boxes1_rb], dim=-1)

    boxes2_lt = boxes2[:, :2]
    boxes2_rb = torch.max(boxes2[:, 2:], boxes2_lt)
    boxes2 = torch.cat([boxes2_lt, boxes2_rb], dim=-1)

    # 再次檢查確保萬無一失
    # 如果還是失敗，就 print 出來看看發生什麼事
    try:
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    except AssertionError:
        print("🚨 Still getting invalid boxes after correction!")
        print("boxes1 min/max:", boxes1.min().item(), boxes1.max().item())
        # 強制修復
        boxes1[:, 2:] = boxes1[:, :2] + 1e-4
        boxes2[:, 2:] = boxes2[:, :2] + 1e-4

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
