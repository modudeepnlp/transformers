class Downloader(object):
    def __init__(self, config: dict):
        self.config = config

    def download(self, download_dir: str):
        raise NotImplementedError
