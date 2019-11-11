# Transformers by Ace
### Easy and Pythonic Workflow on HuggingFace's Transformers
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

Install python
```
pyenv install 3.7.3
pyenv virtualenv 3.7.3 transformers
```

Clone and Create a virtual environment for transformers-ace (main branch: `ace`)
```
git clone https://github.com/modudeepnlp/transformers-ace.git
cd transformers-ace
git remote add upstream https://github.com/huggingface/transformersㄴ
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
``` 
crawl -> CORPUS/ -> prepare -> DATA/train,valid,vocab -> pretrain -> MODELS/
MODELS/ & TASK_DATA/ -> finetune & inference -> TASK_RESULTS/ -> compare -> SUMMARY/
```

### Packages in _transformers_ace_
- `configs`: configs for crawlers, downloaders, pretraining, finetunning
     data for pretraining and tasks from google drive or etc.
- `crawlers`: crawl and save raw text files into `CORPUS/`
- `utils`: utils, downloaders
    - `downloaders`: download corpus, tokenizers, pretrained models,


### Directories
- `CORPUS/`: crawled text data (gs separated format)
- `DATA/`: input data for pretraining (format for each model)
- `TASK_DATA/`: input data for tasks (format for each task)
- `MODELS/`: model files for pytorch
    - (e.g) `MODELS/albert/pretrained/`, `MODELS/albert/fientuned`
- `MODELS_TF/`: model files for tensorflow
- `TASK_RESULTS/`: scores of each task (cola, snli, ...)
- `SUMMARY/`: compare results between models
 
## Instructions
### Preparing
albert with word-piece-tokenizer
```
python -m transformers_ace.albert.prepare --config=word-piece-10000-ko-wiki.json --corpus_dir=~/CORPUS/ko-wiki --data_dir=~/DATA/ko-wiki/wpt
```

### Pretraining
ablert with base-uncased
```
python -m transformers_ace.albert.pretrain --config=albert-base-uncased-config.json --data_dir=~/DATA/ko-wiki/wpt --model_dir=~/MODEL/albert-base-uncased
```

### Finetunning
CoLA task with albert
```
python -m transformers_ace.albert.run_korquad.py --config=korquad_v1.json --model_dir=~/MODEL/bert-base-uncased --result=RESULTS/bert-base-uncased.csv --do_train --do_eval
```

## Crawling (If you need)
```
python -m transformers_ace.crawlers.ko_wiki_crawler --config=ko-wiki-sample.json --corpus_dir=~/CORPUS/ko-wiki
```
