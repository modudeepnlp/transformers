from transformers_ace.temp.albert_pytorch.tools import seed_everything


class Pretrain(object):
    def __init__(self, seed: int):
        seed_everything(seed)
        pass
