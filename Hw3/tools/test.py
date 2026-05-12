# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import zipfile  # 新增：用於產生 zip 壓縮檔
import json  # 新增：用於讀取與排序 JSON
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true")
    # 🌟 新增：讓使用者決定是否要順便跑 validation
    parser.add_argument(
        "--do-val", action="store_true", help="run validation before testing"
    )

    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    cfg.load_from = args.checkpoint

    # =====================================================================
    # 強制設定 Config 中的 Evaluator 輸出路徑
    # 確保不管你的 htc_config 怎麼寫，這裡都會產出我們預期的 JSON 暫存檔
    # =====================================================================
    if "test_evaluator" in cfg:
        outfile_prefix = osp.join(cfg.work_dir, "results_temp")
        if isinstance(cfg.test_evaluator, dict):
            cfg.test_evaluator["format_only"] = True
            cfg.test_evaluator["outfile_prefix"] = outfile_prefix
        elif isinstance(cfg.test_evaluator, list):
            for i in range(len(cfg.test_evaluator)):
                if cfg.test_evaluator[i].get("type") == "CocoMetric":
                    cfg.test_evaluator[i]["format_only"] = True
                    cfg.test_evaluator[i]["outfile_prefix"] = outfile_prefix

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        if "tta_model" not in cfg:
            warnings.warn(
                "Cannot find ``tta_model`` in config, " "we will set it as default."
            )
            cfg.tta_model = dict(
                type="DetTTAModel",
                tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            )
        if "tta_pipeline" not in cfg:
            warnings.warn(
                "Cannot find ``tta_pipeline`` in config, " "we will set it as default."
            )
            test_data_cfg = cfg.test_dataloader.dataset
            while "dataset" in test_data_cfg:
                test_data_cfg = test_data_cfg["dataset"]
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [
                        dict(
                            type="PackDetInputs",
                            meta_keys=(
                                "img_id",
                                "img_path",
                                "ori_shape",
                                "img_shape",
                                "scale_factor",
                                "flip",
                                "flip_direction",
                            ),
                        )
                    ],
                ],
            )
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))

    # =====================================================================
    # 🌟 新增：執行 Validation
    # =====================================================================
    if args.do_val:
        # 檢查 runner 是否有成功載入 val_loop
        if hasattr(runner, "_val_loop") and runner._val_loop is not None:
            print("\n[系統提示] 開始進行 Validation (驗證集評估)...")
            runner.val()
        else:
            print(
                "\n[系統提示] ⚠️ Config 中未設定 Validation 相關參數 (val_dataloader / val_evaluator)，跳過驗證。"
            )

    # =====================================================================
    # 執行 Test
    # =====================================================================
    print("\n[系統提示] 開始進行 Test (測試集推論)...")
    runner.test()

    # =====================================================================
    # 測試完成後，自動進行排序與打包
    # =====================================================================
    print("\n[系統提示] 正在處理作業提交檔案...")

    # CocoMetric 在 format_only 模式下，會生成 .segm.json 和 .bbox.json
    source_json = osp.join(cfg.work_dir, "results_temp.segm.json")
    target_json_name = "test-results.json"
    target_json_path = osp.join(cfg.work_dir, target_json_name)

    # 取得 work-dir 的名稱作為 zip 檔名
    work_dir_name = osp.basename(cfg.work_dir.rstrip("/\\"))
    zip_name = f"{work_dir_name}.zip"
    zip_path = osp.join(cfg.work_dir, zip_name)

    if osp.exists(source_json):
        print("-> 找到預測結果，正在依照 image_id 排序...")
        with open(source_json, "r") as f:
            results_data = json.load(f)

        # 關鍵：依照 image_id 從小到大排序
        results_data.sort(key=lambda x: x["image_id"])

        # 儲存成作業規定的檔名
        with open(target_json_path, "w") as f:
            json.dump(results_data, f)

        # 刪除暫存的未排序檔案
        os.remove(source_json)

        # 順手清理 bbox 的暫存檔，保持資料夾乾淨
        bbox_json = osp.join(cfg.work_dir, "results_temp.bbox.json")
        if osp.exists(bbox_json):
            os.remove(bbox_json)

        print("-> 正在壓縮檔案...")
        # 建立 Zip 檔案 (裡面只會有 test-results.json)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(target_json_path, arcname=target_json_name)

        print(f"✅ 成功！已產生提交檔：{zip_path}")
        print(f"請將此 {zip_name} 檔案直接上傳至 CodaBench！")
    else:
        print(f"❌ 錯誤：找不到輸出的 JSON 檔案 ({source_json})。")
        print("請確保測試資料夾內有圖片，且模型有成功進行預測。")


if __name__ == "__main__":
    main()
