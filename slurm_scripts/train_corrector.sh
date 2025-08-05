PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
DATA_PATH=/mnt/solo/image-corrector

EXP_NAME=code-debug-causal

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --mixed_precision=bf16 \
    --num_processes 3 \
    --num_machines 1 \
    --main_process_port=29506 \
    --dynamo_backend=inductor \
    train_corrector.py \
    --exp-name $EXP_NAME \
    --config $PROJECT_DIR/configs/corrector/corrector.yaml \
    --ar-model-config-path $PROJECT_DIR/configs/randar/randar_l_0.3b_llamagen.yaml \
    --results-dir $DATA_PATH/corrector-results \
    --data-path $DATA_PATH/latents/download/imagenet-llamagen-adm-256_codes \
    --gpt-ckpt $DATA_PATH/pretrain-weights/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors \
    --ckpt-every 250 \
    --num-workers 32