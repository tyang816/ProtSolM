K=20
H=512
pooling_method=mean
CUDA_VISIBLE_DEVICES=1 python run_ft.py \
    --seed 12345 \
    --gnn egnn \
    --gnn_config src/config/egnn.yaml \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --plm facebook/esm2_t33_650M_UR50D \
    --pooling_method $pooling_method \
    --model_dir result/sol/debug/protssn_k"$K"_h"$H" \
    --model_name protssn_"$pooling_method".pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --train_file data/sol/train.csv \
    --valid_file data/sol/valid.csv \
    --test_file data/sol/test.csv \
    --c_alpha_max_neighbors $K \
    --learning_rate 5e-4 \
    --num_train_epochs 10 \
    --batch_token_num 16384 \
    --max_grad_norm 4 \
    --gradient_accumulation_steps 5 \
    --patience 3 \
    --wandb \
    --wandb_project protssn-sol_debug \
    --wandb_run_name protssn_"$pooling_method"_k"$K"_h"$H"_3k