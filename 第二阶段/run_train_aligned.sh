#!/bin/bash
#SBATCH --job-name=pwv_p2_aligned
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --time=3-00:00:00
#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/logs/aligned_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/logs/aligned_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=2451561@tongji.edu.cn

# ============================================================
# 第二阶段: 官方36测试站对齐训练 + 评估 + GPT3基线 (正式脚本)
# ============================================================
source activate kokolo
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export KMP_DUPLICATE_LIB_OK=TRUE

cd /share/home/u23114/tj23114/packages/dachuang_pwv/phase2
mkdir -p logs result_aligned result_gpt3

DATA_DIR=/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data
TEST_STATIONS=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/test_stations_official_36.txt
PY=/share/home/u23114/.conda/envs/kokolo/bin/python

echo "=== 1) 训练 (官方36测试站对齐) ==="
$PY train.py \
    --data_dir "$DATA_DIR" \
    --output_dir result_aligned \
    --batch_size 128 \
    --max_len 30 \
    --epochs 100 \
    --lr 1e-4 \
    --d_model 128 \
    --n_heads 8 \
    --n_layers 4 \
    --ff_dim 512 \
    --dropout 0.1 \
    --patience 20 \
    --test_station_ratio 0.1 \
    --val_ratio 0.15 \
    --seed 42 \
    --num_workers 4 \
    --test_stations "$TEST_STATIONS"

echo "=== 2) 评估 ==="
$PY evaluate.py \
    --data_dir "$DATA_DIR" \
    --model_path result_aligned/best_model.pth \
    --output_dir result_aligned \
    --test_station_ratio 0.1 \
    --val_ratio 0.15 \
    --seed 42 \
    --test_stations "$TEST_STATIONS"

echo "=== 3) GPT3 基线对比 ==="
$PY gpt3_baseline.py \
    --csv result_aligned/test_predictions.csv \
    --grd /share/home/u23114/tj23114/packages/dachuang_pwv/gpt3_1/gpt3_1.grd \
    --out result_gpt3

echo "ALL DONE"
