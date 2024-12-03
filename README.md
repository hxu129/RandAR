# Open Source 🎲 RandAR: Decoder-only Autoregressive Visual Generation in Random Orders

[[`Project Page`](https://rand-ar.github.io/)] [[`arXiv`](https://arxiv.org/abs/2412.01827)] [[`HuggingFace`](https://huggingface.co/ziqipang/RandAR)]

[![arXiv](https://img.shields.io/badge/arXiv-2412.01827-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2412.01827)
[![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://rand-ar.github.io/) 
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat&logo=HuggingFace&logoColor=blue)](https://huggingface.co/ziqipang/RandAR)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## Overview

🎲 **RandAR** is a decoder-only AR model generating image tokens in arbitrary orders. 

🚀 **RandAR** supports parallel-decoding without additional fine-tuning and brings 2.5 $\times$ acceleration for AR generation.

🛠️ **RandAR** unlocks new capabilities for causal GPT-style transformers: inpainting, outpainting, zero-shot resolution extrapolation, and bi-directional feature encoding.

<img src="imgs/teaser.png" alt="teaser" width="100%">

## News

- [12/02/2024] 📋 I am trying my best to re-implement the code and re-train the model as soon as I can. I plan to release the code before 12/09/2024 and the models afterwards. I am going to make my clusters running so fiecely that they will warm up the whole Illinois during this winter. 🔥🔥🔥

- [12/02/2024] 🎉 The paper appears on Arxiv.

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