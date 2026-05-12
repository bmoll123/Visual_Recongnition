# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from .transformer import build_transformer


class DETR(nn.Module):
    """This is the RT-DETR module that performs real-time object detection"""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # 分類與回歸預測頭
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # ========================================================
        # 🌟 RT-DETR 核心：IoU-Aware Query Selection
        # ========================================================
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        # 為了相容性保留預設的 query_embed
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        num_channels = backbone.num_channels
        if isinstance(num_channels, int):
            num_channels = [num_channels]

        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
                for c in num_channels
            ]
        )

        self.backbone = backbone
        self.aux_loss = aux_loss

        # 🌟 綁定 Query Selection 函數
        self.transformer.get_query_selection = self._get_encoder_input

        # ========================================================
        # 🌟 穩定性初始化 (Initialization)
        # ========================================================
        nn.init.xavier_uniform_(self.enc_output.weight)
        nn.init.constant_(self.enc_output.bias, 0)

        # 分類頭 Prior 初始化：防止背景類別在初期造成過大 Loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)
        nn.init.xavier_uniform_(self.class_embed.weight)

    def _get_encoder_input(self, memory):
        bs, seq_len, c = memory.shape

        enc_output = self.enc_output_norm(self.enc_output(memory))
        enc_outputs_class = self.class_embed(enc_output)

        # 🌟 NaN 防護機制：如果出現 NaN，強制賦予極小值，防止 Top-K 崩潰
        max_prob = enc_outputs_class.max(-1)[0]
        max_prob = torch.where(
            torch.isnan(max_prob), torch.full_like(max_prob, -50000.0), max_prob
        )

        topk_proposals = torch.topk(max_prob, self.num_queries, dim=1)[1]

        topk_coords_unact = torch.gather(
            enc_output, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, c)
        )
        return topk_coords_unact

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        pos_embeds = []
        spatial_shapes = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            proj_src = self.input_proj[l](src)
            bs, c, h, w = proj_src.shape
            spatial_shapes.append((h, w))

            srcs.append(proj_src.flatten(2).permute(2, 0, 1))
            masks.append(mask.flatten(1))
            pos_embeds.append(pos[l].flatten(2).permute(2, 0, 1))

        src_concat = torch.cat(srcs, dim=0)
        mask_concat = torch.cat(masks, dim=1)
        pos_concat = torch.cat(pos_embeds, dim=0)

        # 直接呼叫魔改後的 Transformer
        hs = self.transformer(
            src_concat, mask_concat, self.query_embed.weight, pos_concat, spatial_shapes
        )[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR."""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_scores = torch.zeros_like(src_logits)

        if len(idx[0]) > 0:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            ious = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            # 🌟 精度對齊：確保 IoU 分數與 target_scores (FP16) 一致
            target_scores[idx[0], idx[1], target_classes_o] = ious.to(
                target_scores.dtype
            )

        pred_sigmoid = src_logits.sigmoid()
        focal_weight = (
            target_scores * (target_scores > 0.0).float()
            + (1 - target_scores) * (pred_sigmoid**2) * (target_scores == 0.0).float()
        )

        loss_ce = (
            F.binary_cross_entropy_with_logits(
                src_logits, target_scores, reduction="none"
            )
            * focal_weight
        )

        loss_ce = loss_ce * self.empty_weight.view(1, 1, -1)
        losses = {"loss_ce": loss_ce.sum() / num_boxes}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        return self.loss_labels_vfl(outputs, targets, indices, num_boxes, log)

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        pass

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        prob = out_logits.sigmoid()
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]
        return results


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 11 if args.dataset_file != "coco" else 91
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)
    weight_dict = {
        "loss_ce": 1,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    )
    criterion.to(device)

    postprocessors = {"bbox": PostProcess()}
    return model, criterion, postprocessors
