# 1. 繼承基礎的 HTC R50 設定
_base_ = "../configs/htc/htc_r50_fpn_1x_coco.py"

# --- 修改點：將總 Epoch 數改為 50 ---
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)

# --- 修改點：重新設定 50 Epoch 的學習率排程 ---
# 依照比例，將衰減點設在 40 與 46 回合 (約總數的 80% 與 90%)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[40, 46],
        gamma=0.1,
    ),
]

# 2. 定義資料夾路徑與類別
data_root = "/home/yuyun/Desktop/Visual_Recongnition/Hw3/data/"
metainfo = {
    "classes": ("class1", "class2", "class3", "class4"),
    "palette": [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)],
}

# 3. 定義資料讀取管線 (Validation 不需要隨機增強)
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=False),
    dict(type="RandomResize", scale=[(1333, 400), (1333, 900)], keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# 4. 覆寫 DataLoader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train_coco.json",
        data_prefix=dict(img="train/"),
        pipeline=train_pipeline,
    ),
)

# 🌟 修改點：將 Validation 指向 Training Set 🌟
val_cfg = None
val_dataloader = None
val_evaluator = None
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     dataset=dict(
#         type="CocoDataset",
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file="train_coco.json",  # 指向訓練集的標註檔
#         data_prefix=dict(img="train/"),  # 指向訓練集的圖片資料夾
#         test_mode=True,
#         pipeline=test_pipeline,  # 驗證時使用標準 test_pipeline
#     ),
# )

# val_evaluator = dict(
#     type="CocoMetric",
#     ann_file=data_root + "train_coco.json",  # 同樣指向訓練集標註
#     metric=["bbox", "segm"],
# )

# val_cfg = dict(type="ValLoop")

# 5. 測試設定 (用於最終推論)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="test_coco_format.json",
        data_prefix=dict(img="test_release/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "test_coco_format.json",
    metric=["bbox", "segm"],
    format_only=True,
)

test_cfg = dict(type="TestLoop")

# 6. 修改模型 (ResNeXt-101 + PAFPN)
model = dict(
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_32x4d"),
    ),
    neck=dict(
        type="PAFPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    roi_head=dict(
        semantic_head=None,
        semantic_fusion=False,
        bbox_head=[
            dict(type="Shared2FCBBoxHead", num_classes=4),
            dict(type="Shared2FCBBoxHead", num_classes=4),
            dict(type="Shared2FCBBoxHead", num_classes=4),
        ],
        mask_head=[
            dict(type="HTCMaskHead", num_classes=4),
            dict(type="HTCMaskHead", num_classes=4),
            dict(type="HTCMaskHead", num_classes=4),
        ],
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type="soft_nms", iou_threshold=0.5),
            max_per_img=500,
            mask_thr_binary=0.5,
        ),
    ),
)

# 7. 學習率調整
optim_wrapper = dict(optimizer=dict(lr=0.0025))
