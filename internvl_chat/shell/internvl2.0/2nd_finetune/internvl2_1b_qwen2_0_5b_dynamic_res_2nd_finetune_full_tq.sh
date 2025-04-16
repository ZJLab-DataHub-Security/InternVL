set -x

N_NODES=${N_NODES:-4}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS / N_NODES))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full'
LOGGING_DIR='work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full/runs/v1/tf_logs'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 30
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=${N_NODES} \
  --node_rank=0 \
  --master_addr=localhost \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/nas/qianhao/models/InternVL2-1B" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/internvl_1_2_finetune_table_tq.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 0 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_steps 200 \
  --save_total_limit 30 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --recompute_num_llm_layers 24 \
  --recompute_num_vision_layers 24 \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --use_seq_padding True \
  --max_steps -1 \
  --use_llm_compile True \
  --llm_compile_mode "max-autotune-no-cudagraphs" \
  --use_cuda_graph False \
  --cuda_graph_module "decoder_layer" \
  --cuda_graph_layer_num 24 \
  --report_to "tensorboard" \
  --logging_dir ${LOGGING_DIR} \
  2>&1 | tee -a "${LOGGING_DIR}/training_log.txt"

    # --deepspeed "zero_stage1_config.json" \
    # --recompute_num_llm_layers 0 \
    # --recompute_num_vision_layers 0 \
    # --torch_compile_mode "max-autotune" \
    #   --torch_compile_mode "default" \
    # --cuda_graph_module "decoder_layer" \
  # --torch_compile_mode "default" \
  #   --cuda_graph_module "decoder_layer" \
  # --cuda_graph_layer_num 24 \
    # --use_cuda_graph False \
  #     --torch_compile False \
  # --torch_compile_mode "default" \