accelerate launch --mixed_precision=bf16 --multi_gpu \
    train_c2i.py --exp-name randar_0.7b_llamagen_360k \
    --config configs/randar/randar_xl_0.7b_llamagen.yaml \
    --data-path /tmp/imagenet-llamagen-adm-256_codes \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --results-dir /tmp \
    --disk-location /SLOW_DISK/training_ckpts 