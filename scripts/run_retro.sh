#!/bin/bash

name="tune_retro";
export TOKENIZERS_PARALLELISM=false;
mkdir -p results/$name
exec > results/$name/$name.out
exec 2> results/$name/$name.err

# code for training
python main.py \
    --downstream_task 'synthesis' \
    --root 'data/synthesis_data/USPTO_50K_PtoR_aug20' \
    --devices '0,1,2,3' \
    --filename $name \
    --init_checkpoint "all_checkpoints/synthesis_pretrain/last.ckpt" \
    --opt_model $model_name \
    --max_epochs 5 \
    --caption_eval_epoch 50 \
    --check_val_every_n_epoch 1 \
    --save_every_n_epochs 5 \
    --save_ema_checkpoint \
    --save_on_steps \
        22000 \
        24000 \
        26000 \
        28000 \
        30000 \
    --mode ft \
    --disable_graphs \
    --num_beams 20 \
    --num_generate_captions 10 \
    --enable_flash \
    --num_workers 4 \
    --rxn_max_len 512 \
    --text_max_len 512 \
    --llm_tune full \
    --downstream_task synthesis \
    --batch_size 21 \
    --max_inference_len 1024 \
    --accumulate_grad_batches 2 \
    --inference_batch_size 1 \
|| exit 1;

# average the last checkpoints
python average_ckpt.py \
    --checkpoint_paths \
        "all_checkpoints/$name/step_26000_converted.ckpt" \
        "all_checkpoints/$name/step_27000_converted.ckpt" \
        "all_checkpoints/$name/step_28000_converted.ckpt" \
        "all_checkpoints/$name/step_29000_converted.ckpt" \
        "all_checkpoints/$name/step_30000_converted.ckpt" \
    --output_path "all_checkpoints/$name/step_26k-30k.ckpt" \
|| exit 1;

# code for evaluation
python main.py \
    --downstream_task 'synthesis' \
    --root 'data/synthesis_data/USPTO_50K_PtoR_aug20' \
    --devices '0,1' \
    --filename $name \
    --init_checkpoint "all_checkpoints/$name/step_26k-30k.ckpt" \
    --opt_model $model_name \
    --max_epochs 5 \
    --caption_eval_epoch 50 \
    --check_val_every_n_epoch 1 \
    --save_every_n_epochs 5 \
    --mode eval \
    --disable_graphs \
    --num_beams 20 \
    --num_generate_captions 10 \
    --enable_flash \
    --num_workers 4 \
    --rxn_max_len 512 \
    --text_max_len 512 \
    --llm_tune full \
    --downstream_task synthesis \
    --batch_size 32 \
    --max_inference_len 1024 \
    --accumulate_grad_batches 2 \
    --inference_batch_size 1 \
|| exit 1;

# Read the results
echo `date +"%Y-%b-%d %H:%M:%S"` > "results/$name/scores.txt"
for text_file in all_checkpoints/$name/lightning_logs/version_0/*_predictions.txt; do
    base_name=$(basename "$text_file" _predictions.txt)
    python read_results/score.py \
        --beam_size 10 \
        --n_best 10 \
        --augmentation 20 \
        --path $text_file \
        --score_alpha 1 \
        --save_file results/$name/score_$base_name.log \
        > results/$name/score_$base_name.txt ;

    echo -e "\nresults/$name/score_$base_name.txt" >> "results/$name/scores.txt"
    tail -n 2 results/$name/score_$base_name.txt >> "results/$name/scores.txt"
done