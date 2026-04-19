## Train
### Deformable DETR
python main.py \
    --dataset_file coco \
    --coco_path /home/yuyun/Desktop/Visual_Recongnition/Hw2/data \
    --project_name 0420_deformabledetr \
    --batch_size 8 \
    --epochs 100 \
    2>&1 | tee -a results/0420_deformabledetr/terminal_log.txt

