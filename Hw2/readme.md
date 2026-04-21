# NYCU Computer Vision 2026 HW2

* Student ID: 314553034
* Name: 戴郁芸

## Introduciton

* Task Objective: This assignment performs digit detection. The model takes RGB images as input and predicts the correct class and precise bounding box for every digit object in the image.
* Dataset Size:
  * Training Set: 30,062 images
  * Validation Set: 3,340 images
  * Testing Set: 13,068 images

* Core Architecture: Built upon the DETR foundation, I proposed an improved model based on the RT-DETR (Real-Time DEtection TRansformer):
  * Encoder Feature Enhancement: I added a Squeeze-and-Excitation (SE) attention module to the Hybrid Encoder (in CFF module). This uses a channel attention mechanism to adjust the feature weights, helping the model better focus on and capture the key features of the digits.
  * Bounding Box Loss Optimization: I upgraded the Bounding Box Regression Loss from GIoU to CIoU to make the bounding box predictions more accurate.



## Environment Setup

```
conda create -n python3.9 python=3.9 -y

conda activate python3.9

pip install -r requirements.txt
```

## Usage

### Handcraft RTdetr

#### Training
```
python main.py \
    --dataset_file coco \
    --coco_path {DATA_PATH} \
    --project_name {name_of_project} \
    --batch_size 4 \
    --epochs 100 \
    --warmup_steps 2000 \
```

#### Resume training
```
python main.py \
    --dataset_file coco \
    --coco_path {DATA_PATH} \
    --project_name {name_of_project} \
    --resume results/{name_of_project}/checkpoint_last.pth \
    --batch_size 4 \
    --epochs 100 \
    --warmup_steps 2000 \
```

#### Inference
```
python main.py \
    --dataset_file coco \
    --coco_path {DATA_PATH} \
    --project_name {name_of_project} \
    --eval \
    --resume results/{name_of_project}/checkpoint_best.pth
```

------------

### RTdetr

#### Train
```
cd rtdetr_pytorch
python -m torch.distributed.run --nproc_per_node=1 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --amp \
    --seed 42
```

#### Resume training
```
cd rtdetr_pytorch
python -m torch.distributed.run --nproc_per_node=1 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --amp \
    --seed 42 \
    -r output/checkpoint0046.pth
```


#### Inference on testing dataset + Visualize  
```
cd rtdetr_pytorch
python predict.py \
  --rtdetr_dir {location_of_rtdetr_dir} \
  --test_dir {TEST_DATA_PATH} \
  --weights output/rtdetr_r50vd_6x_coco/checkpoint0043.pth \
  --output_dir results \
  --vis_num 20
```


## Performance Snapshot