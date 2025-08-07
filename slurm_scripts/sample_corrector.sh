
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

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 $PROJECT_DIR/sample_corrector.py \
    --exp-name correct-indexing \
    --gpt-ckpt "${DATA_PATH}/pretrain-weights/randar_${MODEL_SIZE}_${TOKENIZER}_360k_bs_1024_lr_0.0004.safetensors" \
    --vq-ckpt $PROJECT_DIR/tokenizer/${TOKENIZER}/vq_ds16_c2i.pt \
    --config "${PROJECT_DIR}/configs/randar/randar_${MODEL_SIZE_LATTER}_${MODEL_SIZE}_${TOKENIZER}.yaml" \
    --corrector-config "${PROJECT_DIR}/configs/corrector/corrector.yaml" \
    --corrector-ckpt "${DATA_PATH}/corrector-results/debug-correct-indexing_bs_288_lr_0.0003/checkpoints/iters_00002000/model.safetensors" \
    --cfg-scales 1.0,3.4 \
    --sample-dir $DATA_PATH/samples \
    --num-inference-steps 88 \
    --num-fid-samples 10000 \
    --per-proc-batch-size 32 \
    --corrector-threshold 0.5 \
    --corrector-max-steps 5