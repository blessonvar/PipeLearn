import config
import math
import torch
import tqdm

from util.timing import Timing
from util.util import get_net, get_data_loader
from util.layer_profiler import Profiling


class Profiler:
    def __init__(self, profiler_config):
        self.config = profiler_config
        self.net = None
        self.optimiser = None
        self.criterion = None
        self.device = None
        self.profiler_timing = Timing('profiler')

    def run(self):
        self.build()
        layers = self.train()
        return layers

    def build(self):
        self.net = get_net(self.config, model_type="complete")
        self.optimiser = torch.optim.SGD(self.net.parameters(), lr=self.config.LEARNING_RATE, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def train(self):
        data_loader = get_data_loader(self.config, set_name='train')
        bar = tqdm.trange(self.config.PROFILE_ITER_NUM)
        self.net.train()
        with Profiling(self.net) as prof:
            for batch_idx in bar:
                inputs, targets = next(iter(data_loader))
                bar.set_description("Profiler iter {}/{}".format(batch_idx + 1, self.config.PROFILE_ITER_NUM))
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # forward pass
                forward_timing = self.profiler_timing.break_down('forward', 0, batch_idx)
                forward_timing.start()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                forward_timing.stop()
                _, predicted = outputs.max(1)
                # backward pass
                backward_timing = self.profiler_timing.break_down('backward', 0, batch_idx)
                backward_timing.start()
                loss.backward()
                backward_timing.stop()
                # param update
                self.optimiser.step()
                self.optimiser.zero_grad()
        return prof.get_results()


class Stage:
    def __init__(self, name, batch_idx, duration):
        """
        A training stage.
        :param name: "server_forward", "server_backward", "client_forward", "client_backward", "uploading" or "downloading"
        :param duration:
        """
        self.name = name
        self.batch_idx = batch_idx
        self.duration = duration
        self.previous_stages = []
        self.next_stages = []

    def add_previous_stage(self, previous_stage):
        self.previous_stages.append(previous_stage)

    def add_next_stage(self, next_stage):
        self.next_stages.append(next_stage)


def recommend_parameter(client_profile_res, server_profile_res, uplink_bandwidth, downlink_bandwidth, layer_num, batch_size):
    min_training_time = None
    best_split_point = None
    best_batch_num = None
    for split_point in range(1, layer_num):
        client_forward = sum([layer['forward'] for layer in client_profile_res][:split_point])
        client_backward = sum([layer['backward'] for layer in client_profile_res][:split_point])
        server_forward = sum([layer['forward'] for layer in server_profile_res][split_point:])
        server_backward = sum([layer['backward'] for layer in server_profile_res][split_point:])
        uploading_time = [layer['output_size'] for layer in client_profile_res][split_point - 1] * 8 / (
                uplink_bandwidth * 1024 * 1024)
        downloading_time = [layer['output_size'] for layer in server_profile_res][split_point - 1] * 8 / (
                downlink_bandwidth * 1024 * 1024)
        batch_num = max(0, math.ceil((uploading_time + server_forward + server_backward + downloading_time) /
                                     (min(client_forward, client_backward) + 1e-8))) + 1
        batch_num = min(min(batch_num, batch_size // 2), 100)
        batch_num = batch_size // (batch_size // batch_num)
        # print("split point: {}\nbatch_num: {}\nclient_forward: {}\nclient_backward: {}\n"
        #       "server_forward: {}\nserver_backward: {}\nupload: {}\ndownload: {}\n".format(
        #         split_point, batch_num, client_forward, client_backward,
        #         server_forward, server_backward, uploading_time, downloading_time))
        training_time = estimate_training_time(
            server_forward, server_backward, client_forward, client_backward,
            uploading_time, downloading_time, batch_num)
        if min_training_time is None or min_training_time > training_time:
            min_training_time = training_time
            best_split_point = split_point
            best_batch_num = batch_num
    # print(best_split_point, best_batch_num)
    return best_split_point, best_batch_num


def estimate_training_time(server_forward, server_backward, client_forward, client_backward,
                           uploading_time, downloading_time, batch_num):
    client_forward_stages = []
    uploading_stages = []
    server_forward_stages = []
    server_backward_stages = []
    downloading_stages = []
    client_backward_stages = []
    # Init stages
    for batch_idx in range(batch_num):
        client_forward_stages.append(Stage("client_forward", batch_idx, client_forward/batch_num))
        uploading_stages.append(Stage("uploading", batch_idx, uploading_time/batch_num))
        server_forward_stages.append(Stage("server_forward", batch_idx, server_forward/batch_num))
        server_backward_stages.append(Stage("server_backward", batch_idx, server_backward/batch_num))
        downloading_stages.append(Stage("downloading", batch_idx, downloading_time/batch_num))
        client_backward_stages.append(Stage("client_backward", batch_idx, client_backward/batch_num))
    # Set up the previous and next stages
    for batch_idx in range(batch_num):
        # client forward stages
        if batch_idx > 0:
            client_forward_stages[batch_idx].add_previous_stage(client_forward_stages[batch_idx-1])
        client_forward_stages[batch_idx].add_next_stage(uploading_stages[batch_idx])
        if batch_idx + 1 < batch_num:
            client_forward_stages[batch_idx].add_next_stage(client_forward_stages[batch_idx+1])
        # uploading stages
        uploading_stages[batch_idx].add_previous_stage(client_forward_stages[batch_idx])
        if batch_idx > 0:
            uploading_stages[batch_idx].add_previous_stage(uploading_stages[batch_idx-1])
        uploading_stages[batch_idx].add_next_stage(server_forward_stages[batch_idx])
        if batch_idx + 1 < batch_num:
            uploading_stages[batch_idx].add_next_stage(uploading_stages[batch_idx+1])
        # server forward and backward stages
        server_forward_stages[batch_idx].add_previous_stage(uploading_stages[batch_idx])
        if batch_idx > 0:
            server_forward_stages[batch_idx].add_previous_stage(server_backward_stages[batch_idx-1])
        server_forward_stages[batch_idx].add_next_stage(server_backward_stages[batch_idx])
        server_backward_stages[batch_idx].add_previous_stage(server_forward_stages[batch_idx])
        server_backward_stages[batch_idx].add_next_stage(downloading_stages[batch_idx])
        if batch_idx + 1 < batch_num:
            server_backward_stages[batch_idx].add_next_stage(server_forward_stages[batch_idx+1])
        # downloading stages
        downloading_stages[batch_idx].add_previous_stage(server_backward_stages[batch_idx])
        if batch_idx > 0:
            downloading_stages[batch_idx].add_previous_stage(server_backward_stages[batch_idx-1])
        downloading_stages[batch_idx].add_next_stage(client_backward_stages[batch_idx])
        if batch_idx + 1 < batch_num:
            downloading_stages[batch_idx].add_next_stage(downloading_stages[batch_idx+1])
        # client backward stages
        client_backward_stages[batch_idx].add_previous_stage(downloading_stages[batch_idx])
        if batch_idx > 0:
            client_backward_stages[batch_idx].add_previous_stage(client_backward_stages[batch_idx-1])
        if batch_idx + 1 < batch_num:
            client_backward_stages[batch_idx].add_next_stage(client_backward_stages[batch_idx+1])
    # estimate training time
    last_stage = client_backward_stages[-1]
    return estimate_time_to_stage(last_stage)


def estimate_time_to_stage(current_stage):
    if current_stage.previous_stages:
        return current_stage.duration + max([estimate_time_to_stage(stage) for stage in current_stage.previous_stages])
    else:
        return current_stage.duration


if __name__ == "__main__":
    profiler_config = config.Config(
            port=11000,
            client_num=4,
            layer_num_on_client=-1,
            epoch_num=1,
            same_data_size_for_each_client=True,
            aggregation_frequency=-1,
            batch_size=100,
            thread_num=4,
            data_size=10000,
            test_data_size=10000,
            model_name='VGG5',
            enable_val=True,
            enable_test=True,
            enable_profiler=False,
            pipe_batch_num=1,
            async_comm=False,
            half_duplex=False,
            duplicate_pipe_batch=False,
            immediate_update=False)
    profiler = Profiler(profiler_config)
    result = profiler.run()
    print(result)
