# April-AE

## Introduction

This repository is a simplified implementation of the paper [&#34;APRIL: Towards Scalable and Transferable Autonomous Penetration Testing in Large Action Space via Action Embedding&#34;](https://ieeexplore.ieee.org/document/10804006).  In this work, we proposed a novel framework to train pentesting agents that are scalable and transferable in large action spaces.

## Related Works

[GAP](https://github.com/Joe-zsc/GAP)

[April](https://github.com/Joe-zsc/April) (comming soon)

## Getting Started

### Installation

Start by checking out the repository:

```bash
git clone https://github.com/Joe-zsc/April-AE.git
cd April-AE
pip install -r requirment.txt
```

### Prepare the embedding models

In this project, we directly use sentence-bert to represent the raw state information and action descriptions as vectors.

1. Download pre-trained [Sentence-BERT](https://huggingface.co/models?library=sentence-transformers) models, or train/fine-tune your own embedding models using domain corpus. (reference: [TSDAE](https://github.com/UKPLab/sentence-transformers))
2. Store the embedding models in path  `NLP_Module\Embedding_models`.
3. Modify the config file `config.ini` and write the model names in the corresponding positions.

```ini
[Embedding]
embedding_models = NLP_Module\Embedding_models
sbert_model = MySbertModel ; your sentence-bert model name, e,g., all-MiniLM-L12-v2
```

4. You can also change the action space size by modifying the config file `config.ini`.

```ini
[common]
...
actions_file = Action-1000 ;Action-5000 and Action-10000 are also avaiable
```

5. Check the simulated training scenarios in `scenarios` file, which are constructed by pre-probing the vulnerable hosts in Vulhub.

### Training with simulated environments

Run the following commands to run a simulation with April-AE:

```bash
python April.py --env_file single\env-CVE-2018-11776.json --agent SAC_AE
```

The learning curves can be seen via the Tensorboard:

```bash
tensorboard --logdir runs --host localhost --port 6666
```

### Training with real vulnerable host

[April](https://github.com/Joe-zsc/April) (comming soon)

## Citation

**NOTE:** This project is for educational purpose only and the author does not condone any illegal use. Use as your own risk.

Please cite our paper at:

```
@ARTICLE{April-AE,
  author={Zhou, Shicheng and Liu, Jingju and Lu, Yuliang and Yang, Jiahai and Hou, Dongdong and Zhang, Yue and Hu, Shulong},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={APRIL: towards Scalable and Transferable Autonomous Penetration Testing in Large Action Space via Action Embedding}, 
  year={2024},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TDSC.2024.3518500}}

```
