class Crawler(object):
    def __init__(self, config: dict):
        self.config = config
        
    def run(self):
        raise NotImplementedError
