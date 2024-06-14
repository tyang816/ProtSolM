K=20
H=512
lr=5e-4
pooling_method=mean
model_name="$pooling_method"_k"$K"_h"$H"_lr"$lr"
project_name=protssn-sol_v2
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/$project_name/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --test_file data/sol/test.csv \
    --test_result_dir result/$project_name/protssn_k"$K"_h"$H"/$model_name \
    --c_alpha_max_neighbors $K \
    --batch_token_num 16000 


K=20
H=512
lr=5e-4
pooling_method=attention1d
model_name=feature_norm_"$pooling_method"_k"$K"_h"$H"_lr"$lr"
project_name=protssn-sol_v2
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/$project_name/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --test_file data/sol/test.csv \
    --test_result_dir result/$project_name/protssn_k"$K"_h"$H"/$model_name \
    --feature_file data/sol/sol_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --c_alpha_max_neighbors $K \
    --batch_token_num 16000


K=20
H=512
lr=5e-4
pooling_method=attention1d
model_name=feature512_norm_pp_"$pooling_method"_k"$K"_h"$H"_lr"$lr"
project_name=protssn-sol_v2
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --seed 3407 \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --model_dir result/$project_name/protssn_k"$K"_h"$H" \
    --model_name $model_name.pt \
    --num_labels 2 \
    --supv_dataset data/sol \
    --test_file data/sol/inden_test.csv \
    --test_result_dir result/$project_name/protssn_k"$K"_h"$H"/$model_name \
    --feature_file data/sol/sol_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --use_plddt_penalty \
    --feature_embed_dim 512 \
    --c_alpha_max_neighbors $K \
    --batch_token_num 16000