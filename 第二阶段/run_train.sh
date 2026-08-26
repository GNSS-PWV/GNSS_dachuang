#!/bin/bash
#SBATCH -J pwv_transformer
#SBATCH -p gpu-a800
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mem=64G

# ============================================================
# 第二阶段 ProfileTransformer 训练 SLURM 脚本
# 适配 A800 / L40 GPU 集群
# ============================================================

set -euo pipefail

# --- 路径配置 (按实际服务器修改) ---
DATA_DIR="/home/user/gnss_pwv/xg_data"        # 训练数据目录 (329 训练站)
OUTPUT_DIR="result"                            # 输出目录
PYTHON="python"                                # 或 conda 环境的 python 路径

# --- 模型超参数 ---
BATCH_SIZE=128
MAX_LEN=30          # 廓线最大层数
EPOCHS=100
LR=1e-4
D_MODEL=128
N_HEADS=8
N_LAYERS=4
FF_DIM=512
DROPOUT=0.1
PATIENCE=20

# --- 数据切分 ---
TEST_RATIO=0.1      # 10% 站点作为测试集
VAL_RATIO=0.15      # 15% 训练数据作为验证集
SEED=42

mkdir -p logs result

echo "========================================"
echo "ProfileTransformer 训练"
echo "  数据: $DATA_DIR"
echo "  输出: $OUTPUT_DIR"
echo "  GPU:  $CUDA_VISIBLE_DEVICES"
echo "  时间: $(date)"
echo "========================================"

$PYTHON train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN \
    --epochs $EPOCHS \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_layers $N_LAYERS \
    --ff_dim $FF_DIM \
    --dropout $DROPOUT \
    --patience $PATIENCE \
    --test_station_ratio $TEST_RATIO \
    --val_ratio $VAL_RATIO \
    --seed $SEED \
    --num_workers 4

echo "训练完成, 开始评估..."
$PYTHON evaluate.py \
    --data_dir "$DATA_DIR" \
    --model_path "$OUTPUT_DIR/best_model.pth" \
    --output_dir "$OUTPUT_DIR" \
    --test_station_ratio $TEST_RATIO \
    --val_ratio $VAL_RATIO \
    --seed $SEED

echo "全部完成: $(date)"