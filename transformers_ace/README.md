# Transformers by Ace
### Easy and Pythonic Workflow based on HuggingFace's Transformers
forked from [huggingface/transformers](https://github.com/huggingface/transformers)

## Install For User (TODO)
```
pip install transformers-ace
```

## Install For Commiter
Install pyenv
```
brew install pyenv
brew install pyenv-virtualenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
```

Install python and create a virtual environment.
```
pyenv install 3.7.3
pyenv virtualenv 3.7.3 transformers
```

Clone this repository.
```
git clone https://github.com/modudeepnlp/transformers-ace.git
cd transformers-ace
pyenv local transformers
```

Install required python packages
```
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r transformers_ace/requirements-dev.txt
```

## Overview
### Data Flow
`crawl` -> CORPUS/sample.csv -> `prepare` -> DATA/sample.txt, tokenizer.model, tokenizer.vocab -> `pretrain` -> MODELS/ 
MODELS/ & TASK_DATA/ -> `finetune` -> TASK_RESULTS/ -> `compare` -> SUMMARY/

### Packages in _transformers_ace_
- `albert`: ALBERT model
- `common`: global config and common classes
- `configs`: configs for downloaders, pretraining, finetunning
     data for pretraining and tasks from google drive or etc.
- `utils`: utils, downloaders
    - `downloaders`: download corpus, tokenizers, pretrained models,
- `etc`: IDE config

### Directories
- `CORPUS/`: crawled text data (gs separated format)
    - `CORPUS_SAMPLE/`: sample data
- `DATA/`: input data for pretraining (format for each model)
- `TASK_DATA/`: input data for tasks (format for each task)
- `MODELS/`: model files for pytorch
    - (e.g) `MODELS/albert/pretrained/`, `MODELS/albert/fientuned`
- `MODELS_TF/`: model files for tensorflow
- `TASK_RESULTS/`: scores of each task (cola, snli, ...)
- `SUMMARY/`: compare results between models
 
## Instructions
### Preparing
Create pretraining data WordPiece Tokenizer ([sentencepiece](https://github.com/google/sentencepiece))
```
python -m transformers_ace.albert.albert_prepare \
    --prepare_config_file=prepare/kowiki-wordpiece-sample.json
```

### Pretraining
Pretrain ALBERT with base-uncased
```
python -m transformers_ace.albert.albert_pretrain \
    --pretrain_config_file=albert/albert-pretrain-sample.json  \ 
    --model_config_file=albert/albert-base-config.json \
    --prepare_config_file=prepare/kowiki-wordpiece-sample.json
```

### Finetunning
Finetune KorQuad task with ALBERT
```
python -m transformers_ace.albert.albert_korquad.py --config_file=configs/finetune/albert-kowiki-wordpiece-sample-korquad1.json
```

## Crawling Corpus
[paul-hyun/web-crawler](https://github.com/paul-hyun/web-crawler)

## Configuration for Python IDEs.
[Pycharm Coding Style like huggingface](https://github.com/modudeepnlp/transformers-ace/blob/ace/transformers_ace/etc/huggingface.xml)
```
Pycharm > Preference > Edit > Code Style > Import Schema > To: "huggingface"
```