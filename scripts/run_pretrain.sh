#!/bin/bash

name="pretrain_hybrid";
export TOKENIZERS_PARALLELISM=false;
mkdir -p results/$name
exec > results/$name/$name.out
exec 2> results/$name/$name.err

python main.py \
    --root 'data/pretrain_data' \
    --devices '0,1' \
    --filename $name \
    --init_checkpoint "all_checkpoints/stage2/last.ckpt" \
    --opt_model 'facebook/galactica-1.3b' \
    --max_epochs 100 \
    --caption_eval_epoch 10000 \
    --save_every_n_epochs 20 \
    --mode pretrain \
    --tune_gnn \
    --enable_flash \
    --num_workers 4 \
    --text_max_len 2048 \
    --pretrain_rxn_num 50000 \
    --context_style "weighted_rxn" \
    --pretrain_use_caption \
    --caption_batch_num 5000 \
    --llm_tune full \
    --batch_size 8 \
    --accumulate_grad_batches 1 \
|| exit 1;

for dir in all_checkpoints/$name/*.ckpt; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir" .ckpt)
        new_name="${base_name}_converted.ckpt"

        python convert.py \
            --input "$dir" \
            --output "all_checkpoints/$name/$new_name" \
        && rm -r "$dir"
    fi
done