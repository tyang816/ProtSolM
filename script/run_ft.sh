K=20
H=512
pooling_method=mean
model_name=protssn_"$pooling_method"_k"$K"_h"$H"
CUDA_VISIBLE_DEVICES=0 python run_ft.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/sol/debug/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --train_file data/sol/train.csv \
    --valid_file data/sol/valid.csv \
    --test_file data/sol/test.csv \
    --c_alpha_max_neighbors $K \
    --learning_rate 5e-4 \
    --num_train_epochs 10 \
    --batch_token_num 25000 \
    --gradient_accumulation_steps 1 \
    --patience 3 \
    --wandb \
    --wandb_project protssn-sol_debug \
    --wandb_run_name $model_name

K=20
H=512
pooling_method=mean
model_name=protssn_feature_"$pooling_method"_k"$K"_h"$H"
CUDA_VISIBLE_DEVICES=5 python run_ft.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/sol/debug/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --train_file data/sol/train.csv \
    --valid_file data/sol/valid.csv \
    --test_file data/sol/test.csv \
    --feature_file data/sol/sol_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --use_plddt_penalty \
    --c_alpha_max_neighbors $K \
    --learning_rate 5e-4 \
    --num_train_epochs 10 \
    --batch_token_num 25000 \
    --gradient_accumulation_steps 1 \
    --patience 3 \
    --wandb \
    --wandb_project protssn-sol_debug \
    --wandb_run_name $model_name