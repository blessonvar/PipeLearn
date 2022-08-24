import os
import sys

import torch
from networks.layer_config import LayerConfig


class Config:
    def __init__(self,
                 experiment_name="experiment",
                 server_address="127.0.0.1",
                 port=51000,
                 client_num=1,
                 layer_num_on_client=-1,
                 same_data_size_for_each_client=True,
                 client_index=0,
                 epoch_num=100,
                 aggregation_frequency=-1,
                 batch_size=100,
                 thread_num=0,
                 data_size=10000,
                 test_data_size=10000,
                 model_name='VGG5',
                 enable_val=False,
                 enable_test=False,
                 enable_profiler=False,
                 pipe_batch_num=1,
                 async_comm=False,
                 half_duplex=False,
                 duplicate_pipe_batch=False,
                 immediate_update=False,
                 profile_iter_num=-1,
                 uplink_bandwidth=50,
                 downlink_bandwidth=50):
        # Framework info.
        self.EXPERIMENT_NAME = experiment_name
        self.SERVER_ADDRESS = server_address
        self.SERVER_PORT = port
        self.CLIENT_NUM = client_num
        self.LAYER_NUM_ON_CLIENT = layer_num_on_client
        self.SAME_DATA_SIZE_FOR_EACH_CLIENT = same_data_size_for_each_client
        self.CLIENT_INDEX = client_index
        self.CONNECTION_TIME_LIMIT = 300
        if thread_num is None or thread_num in [0, -1]:
            thread_num = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
        self.THREAD_NUM = thread_num
        # Dataset info.
        self.DATASET_NAME = 'CIFAR-10'
        self.INPUT_IMAGE_SIZE = 32
        self.NUM_CLASS = 10
        self.DATASET_MEAN = (0.4914, 0.4822, 0.4465)
        self.DATASET_STD = (0.2023, 0.1994, 0.2010)
        # Training info.
        self.DATA_LENGTH = data_size
        self.TEST_DATA_LENGTH = test_data_size
        self.BATCH_SIZE = batch_size
        self.EPOCH_NUM = epoch_num
        self.AGGREGATION_FREQ = aggregation_frequency
        self.LEARNING_RATE = 0.01
        self.CLIENT_DATA_LENGTH = None
        self.CLIENT_TEST_DATA_LENGTH = None
        self.PIPE_BATCH_NUM = pipe_batch_num if layer_num_on_client not in [0, -1] else 1
        if duplicate_pipe_batch:
            self.BATCH_SIZE *= self.PIPE_BATCH_NUM
        self.ASYNC_COMM = async_comm
        self.HALF_DUPLEX = half_duplex
        self.IMMEDIATE_UPDATE = immediate_update
        self.TIME_DIFFERENCE = 0.
        # Model info.
        self.MODEL_NAME = model_name
        self.MODEL_DIR = os.path.join(os.getcwd(), 'models')
        self.MODEL_CONFIG = {
            'VGG5': [
                LayerConfig('Convolutional', in_channel_num=3, out_channel_num=32, kernel_size=3,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=32, out_channel_num=64, kernel_size=3,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=64, out_channel_num=64, kernel_size=3),
                LayerConfig('Dense', in_channel_num=(self.INPUT_IMAGE_SIZE // 2**2)**2*64, out_channel_num=128),
                LayerConfig('Dense', in_channel_num=128, out_channel_num=self.NUM_CLASS)
            ],
            'VGG11': [
                LayerConfig('Convolutional', in_channel_num=3, out_channel_num=64, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=64, out_channel_num=128, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=128, out_channel_num=256, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=256, out_channel_num=256, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=256, out_channel_num=512, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Dense', in_channel_num=(self.INPUT_IMAGE_SIZE // 2**5)**2*512, out_channel_num=4096,
                            drop_out=0.5),
                LayerConfig('Dense', in_channel_num=4096, out_channel_num=4096, drop_out=0.5),
                LayerConfig('Dense', in_channel_num=4096, out_channel_num=self.NUM_CLASS)
            ],
            'VGG16': [
                LayerConfig('Convolutional', in_channel_num=3, out_channel_num=64, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=64, out_channel_num=64, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=64, out_channel_num=128, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=128, out_channel_num=128, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=128, out_channel_num=256, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=256, out_channel_num=256, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=256, out_channel_num=256, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=256, out_channel_num=512, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1),
                LayerConfig('Convolutional', in_channel_num=512, out_channel_num=512, kernel_size=3, padding=1,
                            pool_type='maxpool', pool_kernel_size=2, pool_stride=2),
                LayerConfig('Dense', in_channel_num=(self.INPUT_IMAGE_SIZE//2**5)**2*512, out_channel_num=4096,
                            drop_out=0.5),
                LayerConfig('Dense', in_channel_num=4096, out_channel_num=4096, drop_out=0.5),
                LayerConfig('Dense', in_channel_num=4096, out_channel_num=self.NUM_CLASS)
            ],
            'ResNet18': [
                LayerConfig('Convolutional', in_channel_num=3, out_channel_num=64, kernel_size=7, stride=2, padding=3,
                            bias=False, pool_type='maxpool', pool_kernel_size=3, pool_stride=2, pool_padding=1),
                LayerConfig('ResBasicBlock', in_channel_num=64, out_channel_num=64, stride=1),
                LayerConfig('ResBasicBlock', in_channel_num=64, out_channel_num=64, stride=1),
                LayerConfig('ResBasicBlock', in_channel_num=64, out_channel_num=128, stride=2),
                LayerConfig('ResBasicBlock', in_channel_num=128, out_channel_num=128, stride=1),
                LayerConfig('ResBasicBlock', in_channel_num=128, out_channel_num=256, stride=2),
                LayerConfig('ResBasicBlock', in_channel_num=256, out_channel_num=256, stride=1),
                LayerConfig('ResBasicBlock', in_channel_num=256, out_channel_num=512, stride=2),
                LayerConfig('ResBasicBlock', in_channel_num=512, out_channel_num=512, stride=1,
                            pool_type='avgpool', pool_kernel_size=(1, 1)),
                LayerConfig('Dense', in_channel_num=(self.INPUT_IMAGE_SIZE//2**5)*512, out_channel_num=self.NUM_CLASS)
            ],
            'ResNet50': [
                LayerConfig('Convolutional', in_channel_num=3, out_channel_num=64, kernel_size=7, stride=2, padding=3,
                            bias=False, pool_type='maxpool', pool_kernel_size=3, pool_stride=2, pool_padding=1),
                LayerConfig('ResBottleneck', in_channel_num=64, out_channel_num=64, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=256, out_channel_num=64, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=256, out_channel_num=64, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=256, out_channel_num=128, stride=2),
                LayerConfig('ResBottleneck', in_channel_num=512, out_channel_num=128, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=512, out_channel_num=128, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=512, out_channel_num=128, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=512, out_channel_num=256, stride=2),
                LayerConfig('ResBottleneck', in_channel_num=1024, out_channel_num=256, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=1024, out_channel_num=256, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=1024, out_channel_num=256, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=1024, out_channel_num=256, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=1024, out_channel_num=256, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=1024, out_channel_num=512, stride=2),
                LayerConfig('ResBottleneck', in_channel_num=2048, out_channel_num=512, stride=1),
                LayerConfig('ResBottleneck', in_channel_num=2048, out_channel_num=512, stride=1,
                            pool_type='avgpool', pool_kernel_size=(1, 1)),
                LayerConfig('Dense', in_channel_num=(self.INPUT_IMAGE_SIZE//2**5)*2048, out_channel_num=self.NUM_CLASS)
            ]
        }
        self.LAYER_NUM = len(self.MODEL_CONFIG[self.MODEL_NAME])
        # Result file info.
        self.RESULT_DIR = os.path.join(os.getcwd(), 'results')
        self.RESULT_TXT_ADDRESS = os.path.join(self.RESULT_DIR, 'results.txt')
        self.RESULT_CSV_ADDRESS = os.path.join(self.RESULT_DIR, 'results.csv')
        self.FIG_DIR = os.path.join(self.RESULT_DIR, 'figs')
        # Functions.
        self.ENABLE_VAL = enable_val
        self.ENABLE_TEST = enable_test
        self.MODEL_SAVE = False
        # Profiler
        self.PROFILE_ITER_NUM = profile_iter_num
        self.UPLINK_BANDWIDTH = uplink_bandwidth
        self.DOWNLINK_BANDWIDTH = downlink_bandwidth

    def init_training_info(self):
        if self.SAME_DATA_SIZE_FOR_EACH_CLIENT:
            self.DATA_LENGTH *= self.CLIENT_NUM
        self.CLIENT_DATA_LENGTH = self.DATA_LENGTH // self.CLIENT_NUM
        self.CLIENT_TEST_DATA_LENGTH = self.TEST_DATA_LENGTH // self.CLIENT_NUM
        if self.AGGREGATION_FREQ is None or self.AGGREGATION_FREQ == -1:
            # if AGGREGATION_FREQ is not set, aggregate once each epoch
            self.AGGREGATION_FREQ = self.CLIENT_DATA_LENGTH // self.BATCH_SIZE
        elif self.AGGREGATION_FREQ == 0:
            # if AGGREGATION_FREQ is 0, no aggregation happens.
            self.AGGREGATION_FREQ = sys.maxsize

    def update(self, other, exceptions=None):
        if exceptions is None:
            exceptions = []
        for var_name, var_value in other.__dict__.items():
            if var_name in exceptions:
                continue
            if getattr(self, var_name) != var_value:
                setattr(self, var_name, var_value)
