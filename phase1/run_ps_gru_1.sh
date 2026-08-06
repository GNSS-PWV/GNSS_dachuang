#!/bin/bash
#SBATCH --job-name=ps_train_gru
#SBATCH --comment="train_ps(cnn+bilstm+attention+gru) on L40 GPU"

# 核心资源配置（适配L40集群规则）
# 指定L40分区（请确认集群中L40的分区名，若不是L40需替换为实际名称）
#SBATCH --partition=L40
# 申请GPU数量（L40单卡，适配你的配额需求）
#SBATCH --gres=gpu:l40:1
# 申请CPU核心数（L40单台56核，按单卡7核配比，符合硬件特性）
#SBATCH --cpus-per-task=7
# 作业时长（保持原时长7天，可根据需求调整）
#SBATCH --time=7-00:00:00
# 输出/错误文件路径（保持原路径，方便日志管理）
#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_pwv/PS/%x_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_pwv/PS/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2451561@tongji.edu.cn

# 激活conda环境
source activate kokolo

# GPU环境优化（适配L40和PCIE互联特性）
export CUDA_VISIBLE_DEVICES=0  # 与申请的1卡对应，无需修改
export NCCL_P2P_LEVEL=PCI  # L40是PCIE互联，禁用NVLINK（原A800的NVLINK不适用）
export NCCL_DEBUG=WARN  # 仅输出警告日志，减少冗余
# 新增L40显存优化（单卡48G，适配大显存特性）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 执行训练脚本
python /share/home/u23114/tj23114/packages/dachuang_pwv/PS/PS_train_t.py