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

torchrun --nproc_per_node=4 $PROJECT_DIR/sample_c2i.py \
    --exp-name $MODEL_NAME-test-pipeline \
    --gpt-ckpt "${DATA_PATH}/pretrain-weights/randar_${MODEL_SIZE}_${TOKENIZER}_360k_bs_1024_lr_0.0004.safetensors" \
    --vq-ckpt $PROJECT_DIR/tokenizer/${TOKENIZER}/vq_ds16_c2i.pt \
    --config "${PROJECT_DIR}/configs/randar/randar_${MODEL_SIZE_LATTER}_${MODEL_SIZE}_${TOKENIZER}.yaml" \
    --cfg-scales 1.0,4.0 \
    --sample-dir $DATA_PATH/samples \
    --num-inference-steps 88