from transformers_ace.temp.albert_pytorch.tools import seed_everything


class Prepare(object):
    def __init__(self, seed: int):
        seed_everything(seed)
        pass

    def run(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
