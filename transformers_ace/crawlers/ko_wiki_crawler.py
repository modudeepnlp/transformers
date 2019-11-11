from transformers_ace.crawlers.crawler import Crawler


class KoWikiCrawler(Crawler):
    def __init__(self, config: dict):
        super().__init__(config)

    def run(self):
        pass


if __name__ == '__main__':
    pass
