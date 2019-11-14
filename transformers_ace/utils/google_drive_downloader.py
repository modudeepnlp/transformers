from transformers_ace.utils.downloader import Downloader


class GoogleDriveDownloader(Downloader):
    def __init__(self, url: str):
        super().__init__(url)

    def download(self, download_dir: str):
        raise NotImplementedError
