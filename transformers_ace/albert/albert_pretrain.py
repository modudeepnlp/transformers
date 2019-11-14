""" This referred to https://github.com/lonePatient/albert_pytorch/blob/master/albert_chinese_pytorch/run_pretraining.py """
import os
import time
from argparse import ArgumentParser

import torch
from torch.nn import CrossEntropyLoss

from transformers import AdamW, WarmupLinearSchedule
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_albert import AlbertModel
from transformers.tokenization_albert import AlbertTokenizer
from transformers_ace.albert.albert_prepare import AlbertPrepareConfig
from transformers_ace.common.conf import ACE_ROOT
from transformers_ace.common.pretrain import Pretrain
from transformers_ace.utils.file_util import FileUtil
from transformers_ace.utils.json_util import JsonUtil
from transformers_ace.utils.log_util import LogUtil
from transformers_ace.utils.optimizer_util import OptimizerUtil

log = LogUtil.get_logger(__name__)


class AlbertPretrainConfig(object):
    seed = 1
    gpu_no = -1  # -1=cpu or distributed
    share_type = 'all'  # 'all', 'attention', 'ffn', None=BERT
    epochs = 100
    samples_per_epoch = 1
    eval_steps = 100
    save_steps = 200
    gradient_accumulation_steps = 1
    batch_size = 1
    warmup_proportion = 0.1
    weight_decay = None
    adam_epsilon = 1e-8
    grad_norm = 1.
    learning_rate = 0.001
    fp16_opt_level = None  # '02'  # '00' '01', '02', '03', None


class AlbertPretrain(Pretrain):
    def __init__(self, pretrain_config: AlbertPretrainConfig, prepare_config: AlbertPrepareConfig, model_config: AlbertConfig):
        super(AlbertPretrain, self).__init__(pretrain_config.seed)
        self.pretrain_config = pretrain_config
        self.prepare_config = prepare_config
        self.tokenizer_model_file = f'{prepare_config.model_prefix}.model'
        self.model = AlbertModel(config=model_config)

    def run(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_gpu = torch.cuda.device_count()
        batch_size = self.pretrain_config.batch_size // self.pretrain_config.gradient_accumulation_steps
        total_train_examples = self.pretrain_config.samples_per_epoch * args.epochs
        optimization_steps = int(total_train_examples / batch_size / self.pretrain_config.gradient_accumulation_steps)

        if self.pretrain_config.gpu_no != -1:  # one gpu
            torch.cuda.set_device(self.pretrain_config.gpu_no)
            device = torch.device('cuda', self.pretrain_config.gpu_no)
            n_gpu = 1
            # noinspection PyUnresolvedReferences
            torch.distributed.init_process_group(backend='nccl')
            # noinspection PyUnresolvedReferences
            optimization_steps *= torch.distributed.get_wor

        log.info(f'distributed: {self.pretrain_config.gpu_no != -1}, n_gpu: {n_gpu} (gpu_no: {self.pretrain_config.gpu_no}, device: {device}')

        warmup_steps = int(optimization_steps * self.pretrain_config.warmup_proportion)
        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_model_file)
        model: torch.nn.Module = self.model
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # if args.local_rank != -1:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank) # FIXME:

        optimizer_params = OptimizerUtil.set_weight_decay(list(model.named_parameters()),
                                                          no_decay_params=['bias', 'LayerNorm.bias', 'LayerNorm.weight'],
                                                          weight_decay=self.pretrain_config.weight_decay)
        optimizer = AdamW(optimizer_params, lr=self.pretrain_config.learning_rate, eps=self.pretrain_config.adam_epsilon)
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=optimization_steps)
        epochs = self.pretrain_config.epochs

        if self.pretrain_config.fp16_opt_level is not None:
            try:
                from apex import amp
                model, optimizer = amp.initialize(model, optimizer, opt_level=self.pretrain_config.fp16_opt_level)
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        data_dir = self.prepare_config.data_dir
        self.train(model, data_dir, optimizer, lr_scheduler, epochs, tokenizer, n_gpu)

    def train(self, model, data_dir, optimizer, lr_scheduler, epochs, tokenizer, n_gpu):
        loss = CrossEntropyLoss(ignore_index=-1)

        start_time = time.time()
        for epoch in range(epochs):
            for file in FileUtil.file_list(data_dir):  # TODO: data file list for dynamic masking
                pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pretrain_config_file', default='albert/albert-pretrain-sample.json')
    parser.add_argument('--model_config_file', default='albert/albert-base-config.json')
    parser.add_argument('--prepare_config_file', default='prepare/kowiki-wordpiece-sample.json')
    args = parser.parse_args()

    pretrain_config: AlbertPretrainConfig = JsonUtil.from_json_file(os.path.join(ACE_ROOT, args.config_file))
    prepare_config: AlbertPrepareConfig = JsonUtil.from_json_file(os.path.join(ACE_ROOT, args.prepare_config_file))
    model_config: AlbertConfig = AlbertConfig.from_pretrained(os.path.join(ACE_ROOT, args.model_config_file))
    AlbertPretrain(pretrain_config, prepare_config, model_config).run()
