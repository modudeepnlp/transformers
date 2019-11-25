
## Korquad

Based on the script [`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py).

#### Fine-tuning on Korquad

The data for korquad can be downloaded with the following links and should be saved in a 
'configs' directory.

* KorQuAD_v1.0_train.json
* KorQuAD_v1.0_dev.json
* evaluate-v1.0.py

```bash
python run_squad.py \
  --model_type kobert \
  --model_name_or_path kobert \
  --do_train \
  --do_eval \
  --train_file ../configs/KorQuAD_v1.0_train.json \
  --predict_file ../configs/KorQuAD_v1.0_dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir result/kobert/
```

Training with the previously defined hyper-parameters yields the following results:

```bash
Results: {'exact': 2.234153100103914, 'f1': 4.22841374785126, 'total': 5774}
```
