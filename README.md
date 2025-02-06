# Reinforced Token Optimization (RTO)
This repository contains the source code for our paper [DPO Meets PPO: Reinforced Token Optimization for RLHF](https://arxiv.org/abs/2404.18922). 

Based on theoretical analysis, we propose a more sample efficient and effective RLHF algorithm than PPO (Proximal Policy Optimization). RTO outperforms PPO, DPO (Direct Preference Optimization) and other baselines on AlpacaEval 2 and Arena-Hard benchmarks by a large margin. 

We release all model checkpoints [here](https://huggingface.co/RTO-RL).

## News
- **[2024.4.29]** We released our paper on [arxiv](https://arxiv.org/abs/2404.18922).

## Quick Links
- [Reinforced Token Optimization (RTO)](#reinforced-token-optimization-rto)
  - [News](#news)
  - [Install Requirements](#install-requirements)
  - [Training Scripts](#training-scripts)
  - [Evaluation](#evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Released Models](#released-model)
  - [Citation](#citation)

## Install Requirements
Our code is built on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), therefore we redirect you to their [installation](https://github.com/OpenRLHF/OpenRLHF#installation). Alternatively, you can build from the following scripts:

```bash
conda create -n rto python=3.10
conda activate rto
conda install cuda -c nvidia/label/cuda-12.1.0
pip3 install torch==2.4.1 torchvision torchaudio
cd RTO
pip3 install -e .
```

## Training Scripts
We include the training scripts in `examples/scripts`.
```bash
bash examples/scripts/train_rto_llama_8b.sh
```
This is set for 8xA100 GPUs. You may adjust `micro_rollout_batch_size` and `micro_train_batch_size` based on your computation environment.

## Evaluation
We evaluate on [AlpacaEval 2](https://github.com/tatsu-lab/alpaca_eval) and [Arena-Hard](https://github.com/lm-sys/arena-hard-auto). Please refer to their repository for evaluation.

## Hyperparameter Tuning
Reinforcement learning algorithms may be sensitive to hyperparameter tuning. Based on OpenRLHF's well-tuned hyperparameters for PPO, the only additional parameter to tune is $\beta_1$ (`dpo_reward_scale` in code), the scale of DPO token rewards. Since the main contribution of DPO rewards is reward shaping rather than absolute gains, $\beta_1$ can be safely set to a small value. We recommand using $0.05$ as starting point, but the guideline is not to let DPO token rewards dominate.

## Released Models
We release our models [here](https://huggingface.co/RTO-RL).

| models | AE2 LC | AE2 WR | AH SC | AH WR |
|:---:|--|--|--|--|
| [SFT](https://huggingface.co/OpenRLHF/Llama-3-8b-sft-mixture) | 13.22 | 8.58 | 9.2 | 8.9 |
| [DPO](https://huggingface.co/RTO-RL/Llama3-8B-DPO) | 17.40 | 12.23 | 13.2 | 13.8 |
| [R-DPO](https://huggingface.co/RTO-RL/Llama3-8B-RDPO) | 18.34 | 12.03 | 14.2 | 14.1 |
| [SimPO](https://huggingface.co/RTO-RL/Llama3-8B-SimPO) | 25.46 | 20.20 | 14.5 | 15.2 |
| [TDPO](https://huggingface.co/RTO-RL/Llama3-8B-TDPO) | 20.13 | 11.97 | 13.2 | 12.3 |
| [PPO](https://huggingface.co/RTO-RL/Llama3-8B-PPO) | 19.47 | 12.89 | 16.2 | 15.6 |
| [RTO](https://huggingface.co/RTO-RL/Llama3-8B-RTO) | **27.00** | **22.45** | **20.3** | **21.4** |

## Citation
Please cite our paper if you find the repo helpful in your work:
```bibtex
@misc{zhong2024dpomeetspporeinforced,
      title={DPO Meets PPO: Reinforced Token Optimization for RLHF}, 
      author={Han Zhong and Guhao Feng and Wei Xiong and Xinle Cheng and Li Zhao and Di He and Jiang Bian and Liwei Wang},
      year={2024},
      eprint={2404.18922},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.18922}, 
}
```
