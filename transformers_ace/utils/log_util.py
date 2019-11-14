import logging
import sys
from pathlib import Path


class LogUtil(object):
    @staticmethod
    def get_logger(name='', log_file=None, log_file_level=logging.DEBUG):
        if isinstance(log_file, Path):
            log_file = str(log_file)

        logger = logging.getLogger(name=name)
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)
        if log_file and log_file != '':
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_file_level)
            file_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
            logger.addHandler(file_handler)
        return logger


if __name__ == '__main__':
    log = LogUtil.get_logger(__name__)
    log.debug('hi')
