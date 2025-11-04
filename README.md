# NeuroAda: Activating Each Neuron’s Potential for Parameter-Efficient Fine-Tuning 🚀
This is the official repository for our EMNLP 2025 paper:  [NeuroAda: Activating Each Neuron’s Potential for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2510.18940)

# Environment
git clone https://github.com/FightingFighting/NeuroAda.git
cd NeuroAda
conda env create -f environment.yml

# Dataset
You can download the datasets following [**LLM-Adapters**](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main) or [**Loreft**](https://github.com/stanfordnlp/pyreft/tree/main/examples/loreft).

You can also just use the files in the dataset folder in this repo. They are the same as **LLM-Adapters** and **Loreft**. Download them and unzip them in the dataset folder.

# Train
cd NeuroAda
bash sc

# Result
We provide the **Wandb** link to show our results reported in our paper.
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
| **[LLaMA3 (8B)](https://wandb.ai/z-zhang/NeuroAda/runs/8byjxd6z?nw=nwuserzzhang)** | **0.343%** | 99.7 | 47.8 | 92.7 | 27.6 | 95.7 | 60.4 | 88.7 |  **73.2** |
| **[LLaMA3 (8B)](https://wandb.ai/z-zhang/NeuroAda/runs/n7xpnu0a?nw=nwuserzzhang)** | **0.017%** | 97.2 | 63.7 | 91.9 | 26.4 | 92.9 | 75.0 | 88.7 |  **76.5** |

---

