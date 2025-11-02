# NeuroAda: Activating Each Neuron’s Potential for Parameter-Efficient Fine-Tuning 🚀
This is the official repository for our EMNLP 2025 paper:  [NeuroAda: Activating Each Neuron’s Potential for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2510.18940)



# Result
We provide the wandb link to show our results reported in our paper.
### 🧠 Commonsense Reasoning Results

| 🏗️ Base Model | ⚙️ Params (%) | 🧩 BoolQ | 💡 PIQA | 🤔 SIQA | 📖 HellaS. | 🧍 WinoG. | 🧮 ARC-e | 🧠 ARC-c | 📚 OBQA | 🌟 **Avg.** |
|:--------------:|:-------------:|:--------:|:--------:|:--------:|:-----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **LLaMA (7B)** | **0.404%** | 73.1 | 85.4 | 80.9 | 94.3 | 84.3 | 87.8 | 71.7 | 84.2 |  **82.7** |
| **LLaMA (7B)** | **0.020%** | 69.6 | 83.6 | 80.5 | 92.3 | 81.1 | 84.0 | 67.6 | 84.0 |  **80.0** |
| **LLaMA (13B)** | **0.327%** | 73.3 | 87.9 | 82.7 | 96.0 | 86.9 | 90.2 | 77.1 | 88.6 |  **85.3** |
| **LLaMA (13B)** | **0.016%** | 73.0 | 86.4 | 82.2 | 94.5 | 84.0 | 87.6 | 74.5 | 86.0 |  **83.5** |
| **Llama2 (7B)** | **0.404%** | 73.6 | 86.5 | 81.1 | 94.8 | 87.8 | 89.1 | 75.9 | 85.6 |  **84.3** |
| **Llama2 (7B)** | **0.020%** | 71.4 | 82.8 | 79.8 | 93.3 | 84.0 | 85.4 | 70.1 | 81.2 |  **81.0** |
| **Llama3 (8B)** | **0.343%** | 75.0 | 89.3 | 83.0 | 96.5 | 89.2 | 93.0 | 82.9 | 89.6 |  **87.3** |
| **Llama3 (8B)** | **0.017%** | 71.7 | 84.9 | 81.4 | 93.9 | 85.4 | 88.8 | 77.0 | 83.4 |  **83.3** |

---

### ➗ Arithmetic Reasoning Results

| 🏗️ Base Model | ⚙️ Params (%) | 🔢 MulAri | 📚 GSM8K | ➕ AddSub | 💧 AQuA | 🧮 SinEq | 📊 SVAMP | 📘 MAWPS | 🌟 **Avg.** |
|:--------------:|:-------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **LLaMA (7B)** | **0.404%** | 96.0 | 36.5 | 92.4 | 22.0 | 94.1 | 53.2 | — |  **68.4** |
| **LLaMA (7B)** | **0.020%** | — | 30.3 | — | 22.8 | — | 48.9 | 77.7 |  **44.9** |
| **LLaMA (13B)** | **0.327%** | 97.5 | 43.9 | 92.2 | 21.7 | 93.9 | 61.4 | — |  **71.4** |
| **LLaMA (13B)** | **0.016%** | — | 43.0 | — | 25.6 | — | 56.7 | 83.6 |  **52.2** |
| **LLaMA2 (7B)** | **0.404%** | 97.8 | 39.8 | 91.9 | 20.5 | 96.3 | 54.2 | — |  **70.0** |
| **LLaMA2 (7B)** | **0.020%** | — | 36.1 | — | 22.8 | — | 52.1 | 82.4 |  **48.4** |
| **LLaMA3 (8B)** | **0.343%** | 97.8 | 47.8 | 92.7 | 27.6 | 95.7 | 64.8 | — |  **73.2** |
| **LLaMA3 (8B)** | **0.017%** | — | 63.7 | — | 26.4 | — | 75.0 | 88.7 |  **63.5** |

---




## Commonsense Reasoning
[LLaMA-7B (0.404% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/wtm20rd2?nw=nwuserzzhang)

[LLaMA-7B (0.020% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/1pkira6e?nw=nwuserzzhang)

[LLaMA-13B (0.327% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/00mn0ugz?nw=nwuserzzhang)

[LLaMA-13B (0.016% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/4yin1pcj?nw=nwuserzzhang)

[LLaMA2-7B (0.404% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/dcw5tven?nw=nwuserzzhang)

[LLaMA2-7B (0.020% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/1e9q2svg?nw=nwuserzzhang)

[LLaMA3-8B (0.343% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/w0ua4edu?nw=nwuserzzhang)

[LLaMA3-8B (0.017% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/tk62q1zq?nw=nwuserzzhang)

## Arithmetic Reasoning
[LLaMA-7B (0.404% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/yutx80yi?nw=nwuserzzhang)

[LLaMA-7B (0.020% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/iba9mn5r?nw=nwuserzzhang)

[LLaMA-13B (0.327% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/wr38cxjr?nw=nwuserzzhang)

[LLaMA-13B (0.016% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/zq0h68om?nw=nwuserzzhang)

[LLaMA2-7B (0.404% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/hspdf8kn?nw=nwuserzzhang)

[LLaMA2-7B (0.020% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/psmmv5nr?nw=nwuserzzhang)

[LLaMA3-8B (0.343% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/8byjxd6z?nw=nwuserzzhang)

[LLaMA3-8B (0.017% trainable parameters)](https://wandb.ai/z-zhang/NeuroAda/runs/n7xpnu0a?nw=nwuserzzhang)
