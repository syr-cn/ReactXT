#!/bin/bash

name="tune_action";
export TOKENIZERS_PARALLELISM=false;
mkdir -p results/$name
exec > results/$name/$name.out
exec 2> results/$name/$name.err

python main.py \
    --root 'data/action_data' \
    --devices '0,1' \
    --filename $name \
    --init_checkpoint "all_checkpoints/pretrain_hybrid/last.ckpt" \
    --opt_model 'facebook/galactica-1.3b' \
    --max_epochs 20 \
    --caption_eval_epoch 10 \
    --mode ft \
    --tune_gnn \
    --enable_flash \
    --num_workers 4 \
    --rxn_max_len 512 \
    --text_max_len 512 \
    --downstream_task action \
    --llm_tune full \
    --batch_size 32 \
    --max_inference_len 512 \
    --accumulate_grad_batches 1 \
    --inference_batch_size 16 \
|| exit 1;

cd read_results
python read_results.py \
    --path "../all_checkpoints/$name/lightning_logs/version_0/dataset0_predictions.txt" \
    --name $name;