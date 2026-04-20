# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 🌟 核心修正 1：強制將預測結果轉為 float() (FP32)
        # 這是防止 NaN 的最重要防線，因為 cdist 和 giou 對 FP16 非常敏感
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid().float()
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float()

        # Also concat the target labels and boxes (同樣轉為 float)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).float()

        # 🌟 修改 2：更穩定的 Focal Cost 計算
        # 使用 1e-8 防止 log(0)
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        # 挑選對應標籤的 Cost
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # 🌟 核心修正 2：L1 Cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 🌟 核心修正 3：GIoU Cost (確保輸入是 FP32)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )

        # 🌟 核心修正 4：NaN 最終防護
        # 如果模型不小心噴出 NaN 或 Inf，將它們換成一個很大的數字
        # 這樣匈牙利演算法就不會選到它們，也不會當機
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )
