torchrun tools/extract_latent_codes.py \
    --data-path /tmp/ \
    --code-path /tmp/ \
    --vq-ckpt /tmp/vq_ds16_c2i.pt \
    --config configs/tokenization/llamagen.yaml \
    --image-size 256 \
    --aug-mode adm