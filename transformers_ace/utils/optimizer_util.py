class OptimizerUtil(object):
    # noinspection PyDefaultArgument
    @staticmethod
    def set_weight_decay(params, no_decay_params=[], weight_decay=0.01):
        return [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay_params)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay_params)], 'weight_decay': 0.0}
        ]
