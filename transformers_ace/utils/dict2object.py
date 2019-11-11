class Dict2Object(object):
    def __init__(self, __dic=None):
        if not __dic:
            __dic = {}
        self.__dict__.update(__dic)

    def __repr__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    pass
