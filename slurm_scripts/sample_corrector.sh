
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
    --exp-name $MODEL_NAME-test-pipeline \
    --gpt-ckpt "${DATA_PATH}/pretrain-weights/randar_${MODEL_SIZE}_${TOKENIZER}_360k_bs_1024_lr_0.0004.safetensors" \
    --vq-ckpt $PROJECT_DIR/tokenizer/${TOKENIZER}/vq_ds16_c2i.pt \
    --config "${PROJECT_DIR}/configs/randar/randar_${MODEL_SIZE_LATTER}_${MODEL_SIZE}_${TOKENIZER}.yaml" \
    --corrector-config "${PROJECT_DIR}/configs/corrector/corrector.yaml" \
    --corrector-ckpt "${DATA_PATH}/corrector-results/test-pipeline_bs_1536_lr_0.0001/checkpoints/iters_00001000/model.safetensors" \
    --cfg-scales 1.0,4.0 \
    --sample-dir $DATA_PATH/samples \
    --num-inference-steps 88 \
    --corrector-threshold 0.5 \
    --corrector-max-steps 5