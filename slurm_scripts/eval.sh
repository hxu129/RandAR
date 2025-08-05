PROJECT_DIR=/home/xh/projects/img-corrector/RandAR
MODEL_NAME=randar_0.3b_llamagen_360k
DATA_PATH=/mnt/solo/image-corrector
TOKENIZER=llamagen
MODEL_SIZE=0.3b
if [ $MODEL_SIZE == 0.3b ]; then
    MODEL_SIZE_LATTER=l
else
    MODEL_SIZE_LATTER=xl
fi

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 $PROJECT_DIR/tools/search_cfg_weights.py \
    --config "${PROJECT_DIR}/configs/randar/randar_${MODEL_SIZE_LATTER}_${MODEL_SIZE}_${TOKENIZER}.yaml" \
    --exp-name test-cleanfid-100k \
    --gpt-ckpt "${DATA_PATH}/pretrain-weights/randar_${MODEL_SIZE}_${TOKENIZER}_360k_bs_1024_lr_0.0004.safetensors" \
    --vq-ckpt $PROJECT_DIR/tokenizer/${TOKENIZER}/vq_ds16_c2i.pt \
    --per-proc-batch-size 100 \
    --num-fid-samples-search 1 \
    --num-fid-samples-final 100000 \
    --cfg-scales-interval 0.1 \
    --cfg-scales-search 3.4,3.4 \
    --results-path $DATA_PATH/results \
    --ref-path $DATA_PATH/VIRTUAL_imagenet256_labeled.npz \
    --sample-dir $DATA_PATH/evaluation \
    --num-inference-steps -1