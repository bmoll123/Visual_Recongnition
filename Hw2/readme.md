## Train
### Deformable DETR
python main.py \
    --dataset_file coco \
    --coco_path /home/cvml-3/yy/114_2/Visual_Recongition/Hw2/data \
    --project_name 0420_2_rtdetr \
    --batch_size 8 \
    --epochs 100 \
    --warmup_steps 2000 \
    2>&1 | tee -a results/0420_2_rtdetr/terminal_log.txt

python main.py \
    --dataset_file coco \
    --coco_path /home/cvml-3/yy/114_2/Visual_Recongition/Hw2/data \
    --project_name 0420_2_rtdetr \
    --batch_size 4 \
    --epochs 60 \
    --lr 1e-4 \
    --warmup_steps 2000 \
    2>&1 | tee -a results/0420_2_rtdetr/terminal_log.txt
