# APRIL

We are in the process of organizing the code, and shall release our codes upon acceptance of our paper, for reproduction of the results.

## Installation

Start by checking out the repository:

```bash
git clone https://github.com/Joe-zsc/April-AE.git
cd April-AE
pip install -r requirment.txt
```

## Getting Started

### Prepare the embedding models

1. Download pre-trained [BERT](https://huggingface.co/models) and [Sentence-BERT](https://huggingface.co/models?library=sentence-transformers) models, or train/finetune your own embedding models using domain corpus.
2. Store the embedding models in `NLP_Module\Embedding_models`.
3. Modify the config file `config.ini` and write the model name in the corresponding position.

```ini
[Embedding]
embedding_models=NLP_Module\Embedding_models
bert_model = MyPreTrainedBERT ; your bert model name
sbert_model = all-MiniLM-L12-v2 ; your sentence-bert model name
```

4. You can also change the action space size by modifying the config file `config.ini`.

```ini
[common]
...
actions_file = Action-1000 ;Action-5000 and Action-10000 are also avaiable
```

    5. Check the simulated training scenarios in`scenarios` file, which are constructed by pre-probing the virtualized vulnerable hosts.

## Training

Run the following commands to run a simulation with our proposed RL agent:

```bash
python April.py --env_file single\env-CVE-2018-11776.json --agent SAC_AE
```

The learning curves can be seen via the Tensorboard:

```bash
tensorboard --logdir runs --host localhost --port 6006
```

## Citation
