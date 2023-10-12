import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import logging
from datetime import datetime


def elementwise_apply(fn, packed_sequence):
    '''applies a pointwise function fn to each element in packed_sequence'''
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


def gen_model_name(args):
    return '_'.join([
        datetime.now().strftime('%m%d%H%M'),
        str(args.epochs),
        str(args.bsz),
        str(args.clip),
        f'{args.lr:.1e}'
    ])


class Logger(object):
    DEFAULT_FORMAT = '[%(asctime)s %(src_name)s %(levelname)s] (%(logger_name)s) %(message)s'
    DEFAULT_DATEFMT = '%m-%d %H:%M:%S'

    log_dir = None
    root_logger = None
    model_name = None

    @classmethod
    def config(cls, args, model_name, action='default'):
        cls.model_name = model_name
        cls.log_dir = os.path.join(args.log, model_name, action)

        if not os.path.exists(cls.log_dir):
            os.makedirs(cls.log_dir)

        Logger.root_logger = logging.getLogger()
        Logger.root_logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt=cls.DEFAULT_FORMAT,
            datefmt=cls.DEFAULT_DATEFMT
        )
        # TODO: use `defaults` param when python >= 3.10
        # defaults={
        #     'src_name': '3rd-party lib',
        #     'logger_name': 'default'
        # }

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        Logger.root_logger.addHandler(stream_handler)

        log_path = os.path.join(cls.log_dir, 'console.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        Logger.root_logger.addHandler(file_handler)

        Logger.root_logger = logging.LoggerAdapter(Logger.root_logger, {
            'src_name': 'root',
            'logger_name': 'default'
        })

    def __init__(self, src_name):
        self.loggers = {}
        self.src_name = os.path.split(src_name)[-1]

        self.default_logger = logging.getLogger(self.src_name)
        self.default_logger = logging.LoggerAdapter(
            self.default_logger, {
                'src_name': self.src_name,
                'logger_name': 'default'
            }
        )

    def setup_logger(self, name):
        logger = logging.getLogger(name)

        log_path = os.path.join(self.log_dir, f'{name}.csv')
        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)

        logger = logging.LoggerAdapter(logger, {
            'src_name': self.src_name,
            'logger_name': name
        })

        self.loggers[name] = logger
        return logger

    def get_logger(self, name):
        if name not in self.loggers:
            self.setup_logger(name)
        return self.loggers[name]

    def scalars(self, name, values):
        logger = self.get_logger(name)
        if isinstance(values, (list, tuple)):
            values = ','.join(map(str, values))
        logger.info(str(values))

    def info(self, *messages):
        self.default_logger.info(', '.join(map(str, messages)))
