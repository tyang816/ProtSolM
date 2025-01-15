# ProtSolM: Protein Solubility Prediction with Multi-modal Features

## 🚀 Introduction (Venus-ProtSolM/ ProtSolM)

Fusion of **sequence, structure and feature** information to improve protein solubility prediction!

<img src="img/architecture.png" alt="architecture">

## 📑 Results

### News

- [2024.11.04] I have updated the code and [README file](https://github.com/tyang816/ProtSolM?tab=readme-ov-file#your-own-dataset) to support your own dataset, please have a try!

- [2024.08.22] Congratulations! Our paper was accepted at [*IEEE International Conference on Bioinformatics and Biomedicine 2024 (IEEE BIBM 2024)*](https://ieeexplore.ieee.org/document/10822310)!

### Paper Results

<img src="img/result.png" alt="result">

### Downloads

**PDBSol** and **ExternalTest** pdb files can be found at https://huggingface.co/datasets/tyang816/ProtSolM_ESMFold_PDB.

The labels are stored in CSV files which can be found at `data/PDBSol` and `data/ExternalTest`.

```shell
cd data/PDBSol
wget https://huggingface.co/datasets/tyang816/ProtSolM_ESMFold_PDB/blob/main/ExternalTest_ESMFold_PDB.zip
unzip PDBSol_ESMFold_PDB.zip

cd data/ExternalTest
wget https://huggingface.co/datasets/tyang816/ProtSolM_ESMFold_PDB/blob/main/ExternalTest_ESMFold_PDB.zip
unzip ExternalTest_ESMFold_PDB.zip
```

## 🛫 Requirement

Please make sure you have installed Anaconda3 or Miniconda3.

```shell
conda env create -f environment.yaml
conda activate protsolm
```

## 🧬 Prediction Solubility with ProtSolM

### Download Pre-trained Checkpoints

We use the pre-trained checkpoints from [ProtSSN](https://github.com/tyang816/ProtSSN), we recommend using k20_h512 for fine-tuning on downstream tasks.

| # Version | # Param | # Link                                                       |
| --------- | ------- | ------------------------------------------------------------ |
| k10_h512  | 148     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k10_h512.pt |
| k10_h768  | 160     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k10_h768.pt |
| k10_h1280 | 184     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k10_h1280.pt |
| k20_h512  | 148     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h512.pt |
| k20_h768  | 160     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h768.pt |
| k20_h1280 | 184     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h1280.pt |
| k30_h512  | 148     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k30_h512.pt |
| k30_h768  | 160     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k30_h768.pt |
| k30_h1280 | 184     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k30_h1280.pt |

```shell
mkdir model
cd model
wget https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h512.pt
```

### PDBSol benchmark

#### Extract Features

```shell
python get_feature.py \
    --pdb_dir data/PDBSol/esmfold_pdb \
    --out_file data/PDBSol_feature.csv
```

### Start Testing

Script example can be found at `script/`.

```shell
python eval.py \
    --supv_dataset data/PDBSol \
    --test_file data/PDBSol/test.csv \
    --test_result_dir result/protssn_k20_h512/PDBSol \
    --feature_file data/PDBSol/PDBSol_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --use_plddt_penalty \
    --batch_token_num 3000
```

### Your own dataset

#### What you need at least

- pdb files directory (e.g. `data/<YourDataset>/pdb`).
- a csv file (e.g. `data/<YourDataset>/test.csv`) with the following columns: `name`, `aa_seq`, `label`, if you don't have labels, you can use `0` to replace them.

#### Extract Features

```shell
dataset_name=<YourDataset>
python get_feature.py \
    --pdb_dir data/$dataset_name/pdb \
    --out_file data/$dataset_name/"$dataset_name"_feature.csv
```

#### Start Testing

The result will be saved in `result/$dataset_name`

```shell
python eval.py \
    --supv_dataset data/$dataset_name \
    --test_file data/$dataset_name/test.csv \
    --test_result_dir result/$dataset_name \
    --feature_file data/$dataset_name/"$dataset_name"_feature.csv \
    --feature_name "aa_composition" "gravy" "ss_composition" "hygrogen_bonds" "exposed_res_fraction" "pLDDT" \
    --use_plddt_penalty \
    --batch_token_num 3000
```

### Start Fine-tuning

Script example can be found at `script/`.

```shell
K=20
H=512
pooling_method=attention1d
model_name=feature_"$pooling_method"_k"$K"_h"$H"
CUDA_VISIBLE_DEVICES=0 python run_ft.py \
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
```

## 🙌 Citation

Please cite our work if you have used our code or data. We are pleased to see improvements in the subsequent work.

```
@article{tan2024protsolm,
  title={ProtSolM: Protein Solubility Prediction with Multi-modal Features},
  author={Tan, Yang and Zheng, Jia and Hong, Liang and Zhou, Bingxin},
  journal={arXiv preprint arXiv:2406.19744},
  year={2024}
}
```