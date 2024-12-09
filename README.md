# Open Source 🎲 RandAR: Decoder-only Autoregressive Visual Generation in Random Orders

[[`Project Page`](https://rand-ar.github.io/)] [[`arXiv`](https://arxiv.org/abs/2412.01827)] [[`HuggingFace`](https://huggingface.co/ziqipang/RandAR)]

[![arXiv](https://img.shields.io/badge/arXiv-2412.01827-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2412.01827)
[![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://rand-ar.github.io/) 
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat&logo=HuggingFace&logoColor=blue)](https://huggingface.co/ziqipang/RandAR)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## Overview

Ever thinking about what is the prerequisite for a visual model achieving the impact of GPT in language? The prequisite should be its ability of zero-shot generalization to various applications, prompts, etc. Our RandAR is one of the attempts towards this objective.

🎲 **RandAR** is a decoder-only AR model generating image tokens in arbitrary orders. 

🚀 **RandAR** supports parallel-decoding without additional fine-tuning and brings 2.5 $\times$ acceleration for AR generation.

🛠️ **RandAR** unlocks new capabilities for causal GPT-style transformers: inpainting, outpainting, zero-shot resolution extrapolation, and bi-directional feature encoding.

<img src="imgs/teaser.png" alt="teaser" width="100%">

## News

- [12/09/2024] 🎉 The initial code is released, including the tokenization/modeling/training pipeline. I found that augmentation & tokenization different from the LLaMAGEN's designs are better for FID. From the current speed of training, I expect to release model checkpoints and verified training/eval scripts before 12/18/2024.

- [12/02/2024] 📋 I am trying my best to re-implement the code and re-train the model as soon as I can. I plan to release the code before 12/09/2024 and the models afterwards. I am going to make my clusters running so fiecely that they will warm up the whole Illinois during this winter. 🔥🔥🔥

- [12/02/2024] 🎉 The paper appears on Arxiv.

## Getting Started

Checkout our documentation [DOCUMENTATION.md](documantation.md) for more details.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{pang2024randar,
    title={RandAR: Decoder-only Autoregressive Visual Generation in Random Orders},
    author={Pang, Ziqi and Zhang, Tianyuan and Luan, Fujun and Man, Yunze and Tan, Hao and Zhang, Kai and Freeman, William T. and Wang, Yu-Xiong},
    journal={arXiv preprint arXiv:2412.01827},
    year={2024}
}
```

## Acknowledgement

Thank you to the open-source community for their explorations on autoregressive generation, especially [LLaMAGen](https://github.com/FoundationVision/LlamaGen).