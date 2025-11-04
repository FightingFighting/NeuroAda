# NeuroAda: Activating Each Neuron's Potential for Parameter-Efficient Fine-Tuning 🚀

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202025-blue)](https://arxiv.org/abs/2510.18940)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is the official repository for our EMNLP 2025 paper: **[NeuroAda: Activating Each Neuron's Potential for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2510.18940)**

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation


### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FightingFighting/NeuroAda.git
   cd NeuroAda
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate peft
   ```

## Dataset Preparation

### Option 1: Use Provided Datasets
The repository includes datasets in the `dataset/` folder. These are identical to those used in **LLM-Adapters** and **LoReFT**. Doenload them and unzip them in the folder.

### Option 2: Download Original Datasets
You can download the original datasets from:
- [**LLM-Adapters**](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main)
- [**LoReFT**](https://github.com/stanfordnlp/pyreft/tree/main/examples/loreft)


## Training

### Quick Start
For a basic training run:
```bash
```bash
python train_our.py \
   -task commonsense \
   -data_dir dataset \
   -model yahma/llama-7b-hf \
   -seed 42 \
   -e 3 \
   -lr 7e-4 \
   -batch_size 16 \
   --micro_batch_size 16 \
   -eval_batch_size 16 \
   --test_split test \
   --greedy_decoding \
   --warmup_ratio 0.06 \
   --weight_decay 0 \
   --wandb_project=xxx \
   --wandb_entity=xxx \
   --wandb_watch all \
   --times_num 20 \
   --peft_type perCell_mag_add \
   --max_length 512 \
   --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

```

### Using Pre-configured Scripts
We provide pre-configured training scripts for differen tasks and trainable parameters budget:

#### Commonsense Reasoning
```bash
# LLaMA-7B on commonsense tasks with top-20 paramerters
bash scripts/percell/perCell_mag_add/LLaMA-7B/top20/commonsense.sh

# LLaMA-7B on commonsense tasks with top-1 paramerters
bash scripts/percell/perCell_mag_add/LLaMA-7B/top1/commonsense.sh
```

#### Arithmetic Reasoning
```bash
# LLaMA-7B on commonsense tasks with top-20 paramerters
bash scripts/percell/perCell_mag_add/LLaMA-7B/top20/math.sh

# LLaMA-7B on commonsense tasks with top-1 paramerters
bash scripts/percell/perCell_mag_add/LLaMA-7B/top1/math.sh
```

### Training Parameters

| Parameter | Description | Options |
|-----------|-------------|--------|----------|
| `-task` | Task type | `commonsense`, `math` |
| `-model` | Base model path | `yahma/llama-7b-hf`, `yahma/llama-13b-hf`, `meta-llama/Llama-2-7b-hf`,`meta-llama/Meta-Llama-3-8B` |
| `--peft_type` | PEFT method | `perCell_mag_add` |
| `--target_modules` | Target modules for selecting parameters | See below |
| `--times_num` | Top-K input cionnection for each neuron | `1`, `5`, `10`, `20`, etc.  |
| `-e` | Number of epochs | - |


### Target Modules
Common target modules for different models:
- **LLaMA/LLaMA2/LLaMA3**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Custom selection**: You can specify any subset of these modules



## Results

We provide **Weights & Biases** links to present our results reported in the paper. Below are the results on commonsense and arithmetic reasoning tasks.

### 🧠 Commonsense Reasoning Results

| 🏗️ Base Model | ⚙️ Params (%) | 🧩 BoolQ | 💡 PIQA | 🤔 SIQA | 📖 HellaS. | 🧍 WinoG. | 🧮 ARC-e | 🧠 ARC-c | 📚 OBQA | 🌟 **Avg.** |
|:--------------:|:-------------:|:--------:|:--------:|:--------:|:-----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **[LLaMA (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/wtm20rd2?nw=nwuserzzhang)** | **0.404%** | 73.1 | 85.4 | 80.9 | 94.3 | 84.3 | 87.8 | 71.7 | 84.2 |  **82.7** |
| **[LLaMA (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/1pkira6e?nw=nwuserzzhang)** | **0.020%** | 69.6 | 83.6 | 80.5 | 92.3 | 81.1 | 84.0 | 68.1 | 84.0 |  **80.0** |
| **[LLaMA (13B)](https://wandb.ai/z-zhang/NeuroAda/runs/00mn0ugz?nw=nwuserzzhang)** | **0.327%** | 73.3 | 87.9 | 82.7 | 96.0 | 86.9 | 90.2 | 77.1 | 88.6 |  **85.3** |
| **[LLaMA (13B)](https://wandb.ai/z-zhang/NeuroAda/runs/4yin1pcj?nw=nwuserzzhang)** | **0.016%** | 73.0 | 86.4 | 82.2 | 94.5 | 84.0 | 87.6 | 74.5 | 86.0 |  **83.5** |
| **[Llama2 (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/dcw5tven?nw=nwuserzzhang)** | **0.404%** | 73.6 | 86.5 | 81.1 | 94.8 | 87.8 | 89.1 | 75.9 | 85.6 |  **84.3** |
| **[Llama2 (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/1e9q2svg?nw=nwuserzzhang)** | **0.020%** | 71.4 | 82.8 | 79.8 | 93.3 | 84.0 | 85.4 | 70.1 | 81.2 |  **81.0** |
| **[Llama3 (8B)](https://wandb.ai/z-zhang/NeuroAda/runs/w0ua4edu?nw=nwuserzzhang)** | **0.343%** | 75.0 | 89.3 | 83.0 | 96.5 | 89.2 | 93.0 | 82.9 | 89.6 |  **87.3** |
| **[Llama3 (8B)](https://wandb.ai/z-zhang/NeuroAda/runs/tk62q1zq?nw=nwuserzzhang)** | **0.017%** | 71.7 | 84.9 | 81.4 | 93.9 | 85.4 | 88.8 | 77.0 | 83.8 |  **83.4** |

---

### ➗ Arithmetic Reasoning Results

| 🏗️ Base Model | ⚙️ Params (%) | 🔢 MulAri | 📚 GSM8K | ➕ AddSub | 💧 AQuA | 🧮 SinEq | 📊 SVAMP | 📘 MAWPS | 🌟 **Avg.** |
|:--------------:|:-------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **[LLaMA (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/yutx80yi?nw=nwuserzzhang)** | **0.404%** | 96.0 | 36.5 | 92.4 | 22.0 | 94.1 | 53.2 | 84.5 |  **68.4** |
| **[LLaMA (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/iba9mn5r?nw=nwuserzzhang)** | **0.020%** | 89.0 | 30.3 | 87.1 | 22.8 | 83.7 | 48.9 | 77.7 |  **62.8** |
| **[LLaMA (13B)](https://wandb.ai/z-zhang/NeuroAda/runs/wr38cxjr?nw=nwuserzzhang)** | **0.327%** | 97.5 | 43.9 | 92.2 | 21.7 | 93.9 | 61.4 | 89.1 |  **71.4** |
| **[LLaMA (13B)](https://wandb.ai/z-zhang/NeuroAda/runs/zq0h68om?nw=nwuserzzhang)** | **0.016%** | 94.5 | 43.0 | 88.6 | 25.6 | 90.4 | 56.7 | 83.6 |  **68.9** |
| **[LLaMA2 (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/hspdf8kn?nw=nwuserzzhang)** | **0.404%** | 97.8 | 39.8 | 91.9 | 20.5 | 96.3 | 54.2 | 89.5 |  **70.0** |
| **[LLaMA2 (7B)](https://wandb.ai/z-zhang/NeuroAda/runs/psmmv5nr?nw=nwuserzzhang)** | **0.020%** | 90.8 | 36.1 | 88.4 | 22.8 | 87.6 | 52.1 | 82.4 |  **65.7** |
| **[Llama3 (8B)](https://wandb.ai/z-zhang/NeuroAda/runs/8byjxd6z?nw=nwuserzzhang)** | **0.343%** | 99.7 | 47.8 | 92.7 | 27.6 | 95.7 | 60.4 | 88.7 |  **73.2** |
| **[Llama3 (8B)](https://wandb.ai/z-zhang/NeuroAda/runs/n7xpnu0a?nw=nwuserzzhang)** | **0.017%** | 97.2 | 63.7 | 91.9 | 26.4 | 92.9 | 75.0 | 88.7 |  **76.5** |

---

## Citation

```bibtex
@inproceedings{zhang-etal-2025-neuroada,
    title = "{N}euro{A}da: Activating Each Neuron{'}s Potential for Parameter-Efficient Fine-Tuning",
    author = "Zhang, Zhi  and
      Shen, Yixian  and
      Cao, Congfeng  and
      Shutova, Ekaterina",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.555/",
    pages = "10960--10977",
    ISBN = "979-8-89176-332-6"
}
```

## Acknowledgements

Our code is based on [**LLM-Adapters**](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main) and [**LoReFT**](https://github.com/stanfordnlp/pyreft/tree/main/examples/loreft). We thank the authors for their valuable contributions to the open-source community.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.