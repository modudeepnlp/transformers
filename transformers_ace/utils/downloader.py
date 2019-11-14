class Downloader(object):
    def __init__(self, url: str):
        self.url = url

    def download(self, download_dir: str):
        raise NotImplementedError
