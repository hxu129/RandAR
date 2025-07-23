PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
DATA_PATH=/mnt/solo/image-corrector

accelerate launch --mixed_precision=bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend=inductor \
    train_corrector.py \
    --exp-name test-pipeline \
    --config $PROJECT_DIR/configs/corrector/corrector.yaml \
    --results-dir $DATA_PATH/corrector-results \
    --data-path $PROJECT_DIR/latents/imagenet-llamagen-adm-256_codes \
    --gpt-ckpt $DATA_PATH/pretrain-weights/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors \
    --ar-model-config-path $PROJECT_DIR/configs/randar/randar_l_0.3b_llamagen.yaml \
    --num-workers 16 