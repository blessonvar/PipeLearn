import torch
import time
from functools import reduce
from networks.vgg import Convolutional, Dense
from networks.resnet import ResBasicBlock, ResBottleneck


class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            print("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.record = {'forward': [], 'backward': []}
        self.profiling_on = True
        self.origin_call = {}
        self.hook_done = False
        self.layer_num = 0
        self.instances = [torch.nn.Container, torch.nn.Sequential, Convolutional, Dense, ResBasicBlock, ResBottleneck]
        self.split_points = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)

        self.profiling_on = True
        return self

    def stop(self):
        self.profiling_on = False
        self.recover_modules(self.model)
        return self

    def in_instances(self, this_instance):
        for instance in self.instances:
            if isinstance(this_instance, instance):
                return True
        return False

    def hook_modules(self, module):
        this_profiler = self
        sub_modules = module.__dict__['_modules']
        for i, (name, sub_module) in enumerate(sub_modules.items()):
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break
            if self.in_instances(sub_module):
                self.hook_modules(sub_module)
            else:
                self.layer_num += 1
                split = True
                if i + 1 < len(sub_modules):
                    split = False
                self.split_points.append(split)

                def wrapper_call(self, *input, **kwargs):
                    start_time = time.time()
                    result = this_profiler.origin_call[self.__class__](self, *input, **kwargs)
                    result_size = reduce(lambda x, y: x*y, result.size())
                    stop_time = time.time()

                    if this_profiler.profiling_on:
                        # global record
                        this_profiler.record['forward'].append([self, result_size, start_time, stop_time])

                    that = self

                    def backward_hook(*args):
                        if this_profiler.profiling_on:
                            this_profiler.record['backward'].append([that, result_size, time.time()])

                    result.grad_fn.register_hook(backward_hook)
                    return result

                # Replace "__call__" with "wrapper_call".
                if sub_module.__class__ not in this_profiler.origin_call:
                    this_profiler.origin_call.update({sub_module.__class__: sub_module.__class__.__call__})
                    sub_module.__class__.__call__ = wrapper_call

    def recover_modules(self, module):
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break
            if self.in_instances(sub_module):
                self.recover_modules(sub_module)
            else:
                # Replace "__call__" with "wrapper_call".
                if sub_module.__class__ in self.origin_call:
                    sub_module.__class__.__call__ = self.origin_call[sub_module.__class__]

    def get_results(self):
        # calculate the average time for each layer, including activation layers and batch norm layers.
        iter_num = len(self.record['backward']) // self.layer_num
        for i in range(iter_num):
            backward_start_time = self.record['forward'][(i+1)*self.layer_num-1][3]
            # Update forward time:
            for j in range(self.layer_num):
                self.record['forward'][i*self.layer_num+j][2] = self.record['forward'][i*self.layer_num+j][3] \
                                                                - self.record['forward'][i*self.layer_num+j][2]
                del self.record['forward'][i*self.layer_num+j][3]
            # Update backward time
            for j in range(self.layer_num - 1, -1, -1):
                if j == 0:
                    self.record['backward'][i*self.layer_num+j][2] -= backward_start_time
                else:
                    self.record['backward'][i*self.layer_num+j][2] -= self.record['backward'][i*self.layer_num+j-1][2]
        # print layer info
        # forward = [sum([self.record['forward'][i+j*self.layer_num][2] for j in range(iter_num)]) for i in range(self.layer_num)]
        # backward = [sum([self.record['backward'][i+j*self.layer_num][2] for j in range(iter_num)]) for i in range(self.layer_num)]
        # print("forward: {}".format(forward))
        # print("backward: {}".format(backward))
        # calculate the average time for each layer, excluding activation layers and batch norm layers.
        layer_idx = 0
        layers = []
        # add first layer
        layer_info = {"layer_idx": 0,
                      "name": str(self.record['forward'][0][0]),
                      "output_size": self.record['forward'][0][1],
                      "forward": 0,
                      "backward": 0}
        for i in range(iter_num):
            layer_info['forward'] += self.record['forward'][i * self.layer_num][2] / iter_num
            layer_info['backward'] += self.record['backward'][i * self.layer_num + self.layer_num - 1][2] / iter_num
        split = self.split_points[0]
        # add following layers
        for j in range(1, self.layer_num):
            if split:
                layers.append(layer_info)
                layer_idx += 1
                layer_info = {"layer_idx": layer_idx,
                              "name": str(self.record['forward'][j][0]),
                              "output_size": self.record['forward'][j][2],
                              "forward": 0,
                              "backward": 0}
            for i in range(iter_num):
                layer_info['forward'] += self.record['forward'][i*self.layer_num+j][2] / iter_num
                layer_info['backward'] += self.record['backward'][i*self.layer_num+self.layer_num-j-1][2] / iter_num
            split = self.split_points[j]
        layers.append(layer_info)
        return layers
