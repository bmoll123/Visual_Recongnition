CUDA_VISIBLE_DEVICES=1 python train_ensemble.py --ensemble \
    --ensemble_models resnext101_enhanced resnext101_enhanced\
    --ensemble_weights /mnt/nvme0n1/yuyun/Visual_Recongnition/Hw1/results/0328_resnext101_enhanced_ce_cosine_0.0001_50_64/weights/best.pt /mnt/nvme0n1/yuyun/Visual_Recongnition/Hw1/results/0329_weightedClass_resnext101_enhanced_ce_cosine_0.0001_50_64/weights/best.pt