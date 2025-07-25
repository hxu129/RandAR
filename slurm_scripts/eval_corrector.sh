
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

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=1 $PROJECT_DIR/tools/search_cfg_weights_corrector.py \
    --config "${PROJECT_DIR}/configs/randar/randar_${MODEL_SIZE_LATTER}_${MODEL_SIZE}_${TOKENIZER}.yaml" \
    --exp-name $MODEL_NAME-test-pipeline \
    --gpt-ckpt "${DATA_PATH}/pretrain-weights/randar_${MODEL_SIZE}_${TOKENIZER}_360k_bs_1024_lr_0.0004.safetensors" \
    --vq-ckpt $PROJECT_DIR/tokenizer/${TOKENIZER}/vq_ds16_c2i.pt \
    --per-proc-batch-size 100 \
    --num-fid-samples-search 100 \
    --num-fid-samples-final 50000 \
    --cfg-scales-interval 0.2 \
    --cfg-scales-search 2.0,8.0 \
    --results-path $DATA_PATH/results \
    --ref-path $DATA_PATH/VIRTUAL_imagenet256_labeled.npz \
    --sample-dir $DATA_PATH/evaluation \
    --corrector-config "${PROJECT_DIR}/configs/corrector/corrector.yaml" \
    --corrector-ckpt "${DATA_PATH}/corrector-results/test-pipeline_bs_1536_lr_0.0001/checkpoints/iters_00001000/model.safetensors" \
    --corrector-threshold 0.5 \
    --corrector-max-steps 5 \
    --num-inference-steps 88