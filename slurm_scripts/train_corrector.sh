PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
DATA_PATH=/mnt/solo/image-corrector

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --dynamo_backend=inductor \
    train_corrector.py \
    --exp-name test-transformer \
    --config $PROJECT_DIR/configs/corrector/corrector.yaml \
    --ar-model-config-path $PROJECT_DIR/configs/randar/randar_l_0.3b_llamagen.yaml \
    --results-dir $DATA_PATH/corrector-results \
    --data-path $DATA_PATH/latents/pretrained/imagenet-llamagen-adm-256/imagenet-llamagen-adm-256_codes \
    --gpt-ckpt $DATA_PATH/pretrain-weights/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors \
    --num-workers 16 