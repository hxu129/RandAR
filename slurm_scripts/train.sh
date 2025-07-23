PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
DATA_PATH=/mnt/solo/image-corrector
MODEL_NAME=randar_0.3b_llamagen_360k

accelerate launch --mixed_precision=bf16 --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --dynamo_backend=inductor \
    train_c2i.py \
    --exp-name $MODEL_NAME-test-pipeline \
    --config $PROJECT_DIR/configs/randar/randar_l_0.3b_llamagen.yaml \
    --data-path $PROJECT_DIR/latents/imagenet-llamagen-adm-256_codes \
    --vq-ckpt $PROJECT_DIR/tokenizer/llamagen/vq_ds16_c2i.pt \
    --results-dir $DATA_PATH/results 