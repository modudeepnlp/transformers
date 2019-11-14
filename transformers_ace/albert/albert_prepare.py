import os
from argparse import ArgumentParser

from sentencepiece import SentencePieceTrainer
from tqdm import tqdm

from transformers.tokenization_albert import AlbertTokenizer
from transformers_ace.common.conf import ACE_ROOT
from transformers_ace.common.prepare import Prepare
from transformers_ace.utils.config_util import ConfigUtil
from transformers_ace.utils.file_util import FileUtil
from transformers_ace.utils.log_util import LogUtil

log = LogUtil.get_logger(__name__)


class AlBertPrepareConfig(object):
    seed = 1

    corpus_dir = 'CORPUS_SAMPLE/kowiki/'
    model_prefix = 'MODELS/kowiki/kowiki-wordpiece-sample'
    data_dir = 'DATA/kowiki/kowiki-wordpiece-sample/'
    vocab_size = 10000
    character_coverage = 1.0  # 0.9995 for english, 1.0 for Korean
    model_type = 'unigram'  # 'bpe', 'char', or 'word'
    do_lower_case = True,
    remove_space = True,
    keep_accents = False,
    bos_token = "[CLS]",
    eos_token = "[SEP]",
    unk_token = "<unk>",
    sep_token = "[SEP]",
    pad_token = "<pad>",
    cls_token = "[CLS]",
    mask_token = "[MASK]"


class AlBertPrepare(Prepare):
    def __init__(self, config: AlBertPrepareConfig):
        super().__init__(config.seed)
        self.config = config
        config.corpus_dir = os.path.join(ACE_ROOT, config.corpus_dir)
        config.model_prefix = os.path.join(ACE_ROOT, config.model_prefix)
        config.data_dir = os.path.join(ACE_ROOT, config.data_dir)

        FileUtil.check_and_make_dir(config.corpus_dir)
        FileUtil.check_and_make_dir(os.path.dirname(config.model_prefix))
        FileUtil.check_and_make_dir(config.data_dir)
        assert os.path.isdir(config.corpus_dir), "corpus_dir is not a directory. %s" % config.corpus_dir
        assert os.path.isdir(os.path.dirname(config.model_prefix)), "model_prefix's parent is not a directory. %s" % config.corpus_dir
        assert os.path.isdir(config.data_dir), "data_dir is not a directory. %s" % config.data_dir

    def __create_text(self):
        """
        extract one field from corpus and generate raw text files.
        CORPUS/xxx.csv -> CORPUS/xxx.txt
        """
        for corpus_file in tqdm(FileUtil.file_list(self.config.corpus_dir), desc='create raw text files'):
            if corpus_file.endswith('.csv'):
                text_file = corpus_file.replace('.csv', '.txt')
                if not os.path.isfile(text_file):
                    FileUtil.corpus_file2text_file(corpus_file, text_file, field='text')

    def __create_vocab(self):
        """
        generate pretraining vocab and tokenizer model.
        CORPUS/xxx.txt -> DATA/xxx.vocab, DATA/xxx.model
        """
        for text_file in tqdm(FileUtil.file_list(self.config.corpus_dir), desc='create vocab and tokenizer model'):
            if text_file.endswith('.txt'):
                # """ see: https://github.com/google/sentencepiece#usage-instructions """
                params = f'--input={text_file} --model_prefix={os.path.join(ACE_ROOT, self.config.model_prefix)} --vocab_size={self.config.vocab_size} ' \
                         f'--model_type={self.config.model_type} --character_coverage={self.config.character_coverage}'
                SentencePieceTrainer.Train(params)

    def run(self):
        """
        CORPUS/xxx.txt, DATA/xxx.vocab, DATA/xxx.model -> DATA/xxx.txt
        """
        vocab_file = os.path.join(ACE_ROOT, '%s.vocab' % self.config.model_prefix)
        model_file = os.path.join(ACE_ROOT, '%s.model' % self.config.model_prefix)

        self.__create_text()
        if not os.path.isfile(vocab_file) or not os.path.isfile(model_file):
            self.__create_vocab()

        tokenizer = AlbertTokenizer(vocab_file=vocab_file, model_file=model_file,
                                    do_lower_case=self.config.do_lower_case,
                                    remove_space=self.config.remove_space,
                                    keep_accents=self.config.keep_accents,
                                    bos_token=self.config.bos_token,
                                    eos_token=self.config.eos_token,
                                    unk_token=self.config.unk_token,
                                    sep_token=self.config.sep_token,
                                    pad_token=self.config.pad_token,
                                    cls_token=self.config.cls_token,
                                    mask_token=self.config.mask_token
                                    )
        for text_file in tqdm(FileUtil.file_list(self.config.corpus_dir), desc='create pretraining data files'):
            if text_file.endswith('.txt'):
                data_file = os.path.join(self.config.data_dir, os.path.basename(text_file))
                with open(text_file, 'r') as f, open(data_file, 'w') as fw:
                    for line in f.read().splitlines():
                        tokens = tokenizer.tokenize(line)
                        fw.write(' '.join(tokens) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file', default='configs/kowiki-wordpiece-sample.json')
    args = parser.parse_args()

    config: AlBertPrepareConfig = ConfigUtil.from_json_file(os.path.join(ACE_ROOT, args.config_file))
    AlBertPrepare(config).run()
