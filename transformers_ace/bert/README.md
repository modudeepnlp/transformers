
## Korquad

Based on the script [`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py).

#### Fine-tuning on Koquad

The data for korquad can be downloaded with the following links and should be saved in a 
'configs' directory.

* KorQuAD_v1.0_train.json
* KorQuAD_v1.0_dev.json
* evaluate-v1.0.py

```bash
python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-multilingual-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file ../configs/KorQuAD_v1.0_train.json \
  --predict_file ../configs/KorQuAD_v1.0_dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir result/bert-multilingual/
```

Training with the previously defined hyper-parameters yields the following results:

```bash
Results: {'exact': 18.80845167994458, 
    'f1': 35.42493818385771, 
    'total': 5774}
```
