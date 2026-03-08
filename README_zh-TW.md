# genomic-research

通用基因序列 AI 研究框架。支援多種架構（Transformer、Mamba、CNN、LSTM）和多種任務（預訓練、分類、回歸）。專為 AI agent 自主實驗而設計。

## 特色

- **多種架構**: Transformer（預設）、Mamba SSM、CNN、LSTM，或任何自訂模型
- **多種分詞器**: 字元級、k-mer、BPE
- **多種任務**: 預訓練（MLM/CLM）、序列分類、回歸
- **多種輸入格式**: FASTA、FASTQ、CSV
- **時間預算制**: 固定訓練時間，可重現
- **自動分塊**: 處理任意長度序列（30kb+ 病毒基因組）
- **完整報表**: 訓練曲線、perplexity 圖表、混淆矩陣
- **Agent 友善**: 設計讓 AI agent 自主優化模型

## 快速開始

```bash
pip install genomic-research

# 用 FASTA 序列做預訓練
genomic-research init --fasta sequences.fasta --task pretrain

# 快速測試（30 秒）
GENOMIC_TIME_BUDGET=30 python train.py

# 完整訓練（5 分鐘）
python train.py

# 啟動 AI agent 自主優化
claude
# 然後說："Look at program.md and start experimenting"
```

## 安裝

```bash
# 基本安裝（CPU，僅 Transformer）
pip install genomic-research

# 加 Mamba SSM 支援（需要 CUDA）
pip install genomic-research[mamba]

# 加 BPE 分詞器
pip install genomic-research[bpe]

# 全部安裝
pip install genomic-research[all]
```

## 輸入格式

### FASTA/FASTQ
```bash
genomic-research init --fasta data.fasta --task pretrain
```

### CSV
```bash
# 預訓練
genomic-research init --csv data.csv --seq-col sequence --task pretrain

# 分類
genomic-research init --csv data.csv --seq-col sequence --task classify --label-col species

# 回歸
genomic-research init --csv data.csv --seq-col sequence --task regress --label-col fitness
```

## 架構選項

| 架構 | 複雜度 | 最適用 | 依賴 |
|---|---|---|---|
| Transformer | O(n²) | 通用、中等長度序列 | 純 PyTorch |
| Mamba | O(n) | 長序列（>1kb）、高效預訓練 | `mamba-ssm`（CUDA） |
| CNN | O(n) | 局部模式偵測、快速訓練 | 純 PyTorch |
| LSTM | O(n) | 序列模式 | 純 PyTorch |

## 分詞器

| 分詞器 | 詞彙量 | tokens/bp | 最適用 |
|---|---|---|---|
| `char` | 10 | 1.0 | Mamba/SSM、簡單基線 |
| `kmer` (k=6) | 4101 | ~0.17 | Transformer（壓縮序列 6 倍） |
| `bpe` | 可設定 | 可變 | 大型資料集、最佳壓縮 |

## 運作原理

```
你的序列 (.fasta/.fastq/.csv)
    │
    ▼
prepare.py  ──►  分詞 + 分塊 + 分割  ──►  ~/.cache/genomic-research/
    │                                                    │
    │  （固定，不修改）                                    │
    ▼                                                    ▼
train.py  ──►  模型 + 訓練迴圈  ──►  reports/
    │
    │  （AI agent 修改這個檔案）
    ▼
評估 & 生成報表
```

1. **prepare.py**（固定）：載入序列、分詞、分塊、分割訓練/驗證集
2. **train.py**（可修改）：定義模型架構和訓練迴圈，AI agent 修改此檔案
3. **program.md**：AI agent 的實驗指令

## 搭配 AI Agent 使用

```bash
# 1. 初始化
genomic-research init --fasta viral_genomes.fasta --task pretrain

# 2. 啟動 AI agent
claude

# 3. 給 agent 指令
"Look at program.md and start experimenting"
```

Agent 會自動：
- 閱讀 `program.md` 了解實驗流程
- 建立 git 分支
- 修改 `train.py` 嘗試不同架構和超參數
- 執行實驗並追蹤結果到 `results.tsv`
- 保留改進、捨棄失敗

## 授權

MIT
