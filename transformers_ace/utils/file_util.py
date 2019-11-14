import csv
import os
import sys

import pandas
from tqdm import tqdm

from transformers_ace.common.conf import FIELD_SEPARATOR


class FileUtil(object):
    @staticmethod
    def file_list(path):
        corpus_files = []
        for _file_or_dir in os.listdir(path):
            file_or_dir = os.path.join(path, _file_or_dir)
            if os.path.isfile(file_or_dir) and not file_or_dir.startswith('.'):
                corpus_files.append(os.path.join(path, file_or_dir))
        return corpus_files

    @staticmethod
    def check_and_make_dir(path):
        if not os.path.isdir(path):
            os.makedirs(path)

    @staticmethod
    def corpus_file2text_file(corpus_file, text_file, field) -> str:
        csv.field_size_limit(sys.maxsize)
        df = pandas.read_csv(corpus_file, sep=FIELD_SEPARATOR, engine='python')
        lines = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='read data'):
            lines.append(row[field])

        with open(text_file, 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    FileUtil.corpus_file2text_file('/Users/bage/workspace/transformers-ace/transformers_ace/CORPUS_SAMPLE/kowiki/sample.csv', 'text')
