#!/bin/bash


#SBATCH --partition=gpu_h100
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=llama-7b-hf
#SBATCH --time=20:00:00
#SBATCH --output=./output/slurm_output/%A_%a.out
#SBATCH --array=0


learning_rates=(7e-4)
batch_sizes=(16)
times_nums=(20)
weight_decays=(0)
warmup_ratios=(0.06)
num_epochs=(3)

micro_batch_size=16

for lr_ in "${learning_rates[@]}"; do
    for batch_size_ in "${batch_sizes[@]}"; do
        for times_num_ in "${times_nums[@]}"; do
          for weight_decay_ in "${weight_decays[@]}"; do
            for warmup_ratio_ in "${warmup_ratios[@]}"; do
              for num_epoch_ in "${num_epochs[@]}"; do
                  combinations+=("$lr_;$batch_size_;$times_num_;$weight_decay_;$warmup_ratio_;$num_epoch_")
              done
            done
          done
        done
    done
done
selected_combination=${combinations[$SLURM_ARRAY_TASK_ID]}
IFS=';' read -r lr batch_size times_num  weight_decay warmup_ratio num_epoch <<< "$selected_combination"




source activate peft


model_name=yahma/llama-7b-hf
seed=42
max_seq_length=512
peft_type=perCell_mag_add
task=commonsense
project_name=NeuroAda


target_modules=("q_proj" "k_proj" "v_proj" "o_proj"  "gate_proj" "up_proj" "down_proj")  # "o_proj"  "gate_proj"

joined_str=$(IFS="_"; echo "${target_modules[*]}")


job_id=$SLURM_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID

entity_name=${job_id}_${task_id}_$(echo "$model_name" | sed 's/\//-/g')_${task}_${peft_type}_times_num${times_num}_batch_size${batch_size}_lr${lr}_num_epoch${num_epoch}_target_modules_${joined_str}_weight_decay${weight_decay}_warmup_ratio${warmup_ratio}_max_seq_length${max_seq_length}
echo $entity_name


python train_our.py -task $task \
-data_dir dataset \
-model $model_name \
-seed $seed -e $num_epochs -lr $lr \
-batch_size $batch_size \
--micro_batch_size $micro_batch_size \
-eval_batch_size 16 \
--test_split test \
--greedy_decoding \
--warmup_ratio $warmup_ratio \
--weight_decay $weight_decay \
--wandb_project=$project_name \
--wandb_entity=$entity_name \
--wandb_watch all \
--times_num $times_num \
--target_modules "${target_modules[@]}" \
--peft_type $peft_type \
--max_length $max_seq_length \
#--max_n_train_example 500 \




