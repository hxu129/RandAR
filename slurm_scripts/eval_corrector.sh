
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

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29503 $PROJECT_DIR/tools/search_cfg_weights_corrector.py \
    --config "${PROJECT_DIR}/configs/randar/randar_${MODEL_SIZE_LATTER}_${MODEL_SIZE}_${TOKENIZER}.yaml" \
    --exp-name eval-correct-indexing \
    --gpt-ckpt "${DATA_PATH}/pretrain-weights/randar_${MODEL_SIZE}_${TOKENIZER}_360k_bs_1024_lr_0.0004.safetensors" \
    --vq-ckpt $PROJECT_DIR/tokenizer/${TOKENIZER}/vq_ds16_c2i.pt \
    --per-proc-batch-size 32 \
    --num-fid-samples-search 1000 \
    --num-fid-samples-final 10000 \
    --cfg-scales-interval 0.0 \
    --cfg-scales-search 3.4,3.4 \
    --results-path $DATA_PATH/results \
    --ref-path $DATA_PATH/VIRTUAL_imagenet256_labeled.npz \
    --sample-dir $DATA_PATH/evaluation \
    --corrector-config "${PROJECT_DIR}/configs/corrector/corrector.yaml" \
    --corrector-ckpt "${DATA_PATH}/corrector-results/debug-correct-indexing_bs_288_lr_0.0003/checkpoints/iters_00001500/model.safetensors" \
    --corrector-threshold 0.8 \
    --corrector-max-steps 5 \
    --num-inference-steps 88
