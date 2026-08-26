#!/bin/bash
#SBATCH --job-name=gru_ts_tm_train_a800
#SBATCH --comment="train_gru_ts_tm(gru) on A800 GPU"

# 核心资源配置（适配A800集群规则）
# 指定A800分区（请确认集群中A800的分区名，若不是A800需替换为实际名称）
#SBATCH --partition=A800
# 申请GPU数量（A800单卡，适配你的配额需求）
#SBATCH --gres=gpu:a800:1
# 申请CPU核心数（A800单台40核，按单卡7核配比，符合硬件特性）
#SBATCH --cpus-per-task=7
# 作业时长（保持原时长7天，可根据需求调整）
#SBATCH --time=7-00:00:00
# 输出/错误文件路径（保持原路径，方便日志管理）
#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_pwv/TS_TM/%x_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_pwv/TS_TM/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2451561@tongji.edu.cn

# 激活conda环境
source activate kokolo

# GPU环境优化（适配A800和NVLINK互联特性）
export CUDA_VISIBLE_DEVICES=0  # 与申请的1卡对应，无需修改
export NCCL_P2P_LEVEL=NVL  # A800支持NVLINK，启用NVLINK加速
export NCCL_DEBUG=WARN  # 仅输出警告日志，减少冗余
# A800显存优化（单卡80G，适配大显存特性）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# 执行训练脚本
python /share/home/u23114/tj23114/packages/dachuang_pwv/TS_TM/gru_ts_tm.py
