PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
DATA_PATH=/mnt/solo/ImageNet/

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 $PROJECT_DIR/tools/extract_latent_codes.py \
    --data-path $DATA_PATH \
    --code-path $DATA_PATH/latents/pretrained/imagenet-llamagen-adm-256/val \
    --tokenizer-name llamagen \
    --vq-ckpt $PROJECT_DIR/tokenizer/llamagen/vq_ds16_c2i.pt \
    --config $PROJECT_DIR/configs/tokenization/llamagen.yaml \
    --image-size 256 \
    --aug-mode adm \
    --num-workers 16 \
    --global-seed 0 \
    --debug