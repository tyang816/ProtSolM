#!/bin/bash
#SBATCH -J sol
#SBATCH -p NvidiaA800
#SBATCH --nodelist=ZSGPU10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=180-00:00:00
#SBATCH --output=/public/home/tanyang/workspace/sol/log/%j.out
#SBATCH --error=/public/home/tanyang/workspace/sol/log/%j.err

source ~/.zshrc
zsh
conda activate protssn
cd /public/home/tanyang/workspace/sol
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export OMP_PROC_BIND=false

pooling_method=attention1d
lr=5e-4
model_name=feature_norm_"$pooling_method"_k"$K"_h"$H"_lr"$lr"
project_name=protssn-sol_v2
#     --feature_embed_dim 512 \

CUDA_VISIBLE_DEVICES=0 python run_ft.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/$project_name/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --train_file data/sol/train.csv \
    --valid_file data/sol/valid.csv \
    --test_file data/sol/test.csv \
    --feature_file data/sol/sol_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --c_alpha_max_neighbors $K \
    --learning_rate $lr \
    --num_train_epochs 30 \
    --batch_token_num 16000 \
    --gradient_accumulation_steps 1 \
    --patience 5 \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project $project_name \
    --wandb_run_name $model_name