PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
torchrun --nproc_per_node=4 $PROJECT_DIR/tools/extract_latent_codes.py \
    --data-path /mnt/solo/ImageNet/ \
    --code-path $PROJECT_DIR/latents \
    --tokenizer-name llamagen \
    --vq-ckpt $PROJECT_DIR/tokenizer/llamagen/vq_ds16_c2i.pt \
    --config $PROJECT_DIR/configs/tokenization/llamagen.yaml \
    --image-size 256 \
    --aug-mode adm \
    --num-workers 16 \
    --global-seed 0