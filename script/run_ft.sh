K=20
H=512
lr=5e-4
pooling_method=mean
model_name="$pooling_method"_k"$K"_h"$H"_lr"$lr"
project_name=protssn-sol_v2
CUDA_VISIBLE_DEVICES=0 python run_ft.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/$project_name/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/PDBSol \
    --train_file data/PDBSol/train.csv \
    --valid_file data/PDBSol/valid.csv \
    --test_file data/PDBSol/test.csv \
    --c_alpha_max_neighbors $K \
    --learning_rate 5e-4 \
    --num_train_epochs 30 \
    --batch_token_num 16000 \
    --gradient_accumulation_steps 1 \
    --patience 5 \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project $project_name \
    --wandb_run_name $model_name

K=20
H=512
pooling_method=attention1d
model_name=feature_"$pooling_method"_k"$K"_h"$H"
CUDA_VISIBLE_DEVICES=1 python run_ft.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/sol/debug/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/PDBSol \
    --train_file data/PDBSol/train.csv \
    --valid_file data/PDBSol/valid.csv \
    --test_file data/PDBSol/test.csv \
    --feature_file data/PDBSol/PDBSol_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --c_alpha_max_neighbors $K \
    --learning_rate 5e-4 \
    --num_train_epochs 10 \
    --batch_token_num 16000 \
    --gradient_accumulation_steps 1 \
    --patience 3 \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project protssn-sol_debug \
    --wandb_run_name $model_name

for i in {9065..9076}
do
    scancel $i
done