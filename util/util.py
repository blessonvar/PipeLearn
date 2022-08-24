import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from networks.vgg import VGG
from networks.resnet import ResNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_net(config, model_type="server"):
    """
    Build a neural network
    :param config: network config
    :param model_type: "complete": complete model; "server": server-side model; "client": client-side model.
    """
    if config.MODEL_NAME.startswith("VGG"):
        net = VGG(config, model_type)
    elif config.MODEL_NAME.startswith("ResNet"):
        net = ResNet(config, model_type)
    else:
        raise Exception("Undefined Neural Network.")
    logger.debug(str(net))
    return net


def get_data_loader(config, set_name='train'):
    home = os.getcwd()
    dataset_path = os.path.join(home, 'data', config.DATASET_NAME)
    data_path = os.path.join(dataset_path, 'train') if set_name == 'train' else os.path.join(dataset_path, 'test')
    download = True if not os.path.exists(data_path) or not os.listdir(data_path) else False
    data_length = config.DATA_LENGTH if set_name == 'train' else config.TEST_DATA_LENGTH
    client_data_length = config.CLIENT_DATA_LENGTH if set_name == 'train' else config.CLIENT_TEST_DATA_LENGTH
    indices = list(range(data_length))
    local_data_indices = indices[
        client_data_length * config.CLIENT_INDEX:
        client_data_length * (config.CLIENT_INDEX + 1)
    ] if set_name == 'train' else indices[
        :client_data_length
    ]
    if set_name == 'train':
        transform = transforms.Compose([
            transforms.Resize(config.INPUT_IMAGE_SIZE),
            transforms.RandomCrop(config.INPUT_IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize(config.INPUT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        shuffle = False
    dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=set_name == 'train', download=download, transform=transform)
    subset = Subset(dataset, local_data_indices)
    if set_name == 'train':
        train_data_loader = DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
        return train_data_loader
    else:
        if config.ENABLE_VAL:
            val_length = int(client_data_length * 0.2)
            test_length = client_data_length - val_length
            val_set, test_set = random_split(subset, [val_length, test_length])
            return DataLoader(val_set if set_name == 'val' else test_set, batch_size=config.BATCH_SIZE, shuffle=shuffle)
        else:
            return None if set_name == 'val' else DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=shuffle)


def get_average_model(models, weights=None):
    if weights is None or weights == []:
        weights = [1 for _ in models]
    assert len(weights) == len(models), \
        "The number of weights {} is not consistent with the number of models {}.".format(len(weights), len(models))
    return {
        weight_name: sum([models[i][weight_name] * weights[i] for i in range(len(models))]) / sum(weights)
        for weight_name in models[0].keys()
    }


def get_optimiser(net, layer_num_on_client, learning_rate):
    if (layer_num_on_client == -1 and net.model_type == 'server') or (
            layer_num_on_client == 0 and net.model_type == 'client'):
        return None
    return torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


def get_state_dict_size(state_dict):
    total_size = 0
    for params in state_dict.values():
        param_size = 0
        for dim_size in params.size():
            if param_size == 0:
                param_size = dim_size
            else:
                param_size *= dim_size
        total_size += param_size
    return total_size


def print_log(config, test_accuracy, training_time, communication_amount, split_points, parallel_batch_nums):
    logger.info("Experiment configuration:")
    logger.info("Client number: {}".format(config.CLIENT_NUM))
    logger.info("Layer number on clients: {}".format(split_points))
    logger.info("Parallel batch number: {}\n".format(parallel_batch_nums))
    logger.info("Whether same data size for each client: {}".format(config.SAME_DATA_SIZE_FOR_EACH_CLIENT))
    logger.info("Data size: {}".format(config.DATA_LENGTH))
    logger.info("Batch size: {}".format(config.BATCH_SIZE))
    logger.info("Aggregation frequency: {}".format(config.AGGREGATION_FREQ))
    logger.info("Epoch number: {}".format(config.EPOCH_NUM))
    logger.info("Thread number: {}".format(config.THREAD_NUM))
    logger.info("Results:")
    logger.info('Test Accuracy: {:.2f}'.format(test_accuracy))
    logger.info('Training time: {:.2f}'.format(training_time))
    logger.info('Communication amount in training: {:.4e}'.format(communication_amount.get('TRAINING', 0)))
    logger.info('Communication amount in aggregation: {:.4e}'.format(communication_amount.get('AGGREGATION', 0)))
    logger.info('Communication amount in testing: {:.4e}'.format(communication_amount.get('TESTING', 0)))
    logger.info('Total communication amount: {:.4e}'.format(communication_amount['TOTAL']))


def save_result(config, file, test_accuracy, training_time, communication_amount, split_points, parallel_batch_nums):
    file.write("Experiment configuration:\n")
    file.write("Client number: {}\n".format(config.CLIENT_NUM))
    file.write("Layer number on clients: {}\n".format(split_points))
    file.write("Parallel batch number: {}\n".format(parallel_batch_nums))
    file.write("Whether same data size for each client: {}\n".format(config.SAME_DATA_SIZE_FOR_EACH_CLIENT))
    file.write("Data size: {}\n".format(config.DATA_LENGTH))
    file.write("Batch size: {}\n".format(config.BATCH_SIZE))
    file.write("Aggregation frequency: {}\n".format(config.AGGREGATION_FREQ))
    file.write("Epoch number: {}\n".format(config.EPOCH_NUM))
    file.write("Tread number: {}\n".format(config.THREAD_NUM))
    file.write("Results:\n")
    file.write("Test accuracy: {:.2f}\n".format(test_accuracy))
    file.write("Training time: {:.2f}\n".format(training_time))
    file.write("Training communication amount: {:.4e}\n".format(communication_amount.get("TRAINING", 0)))
    file.write("Aggregation communication amount: {:.4e}\n".format(communication_amount.get("AGGREGATION", 0)))
    file.write("Testing communication amount: {:.4e}\n".format(communication_amount.get("TESTING", 0)))
    file.write("Total communication amount: {:.4e}\n".format(communication_amount["TOTAL"]))
    file.write("\n")


def split_batch(inputs, targets, partition_num=1):
    chunk_size = inputs.size(dim=0) // partition_num
    micro_inputs = inputs.split(chunk_size, 0)
    micro_targets = targets.split(chunk_size, 0)
    if inputs.size(dim=0) % chunk_size != 0:
        return zip(micro_inputs[:-1], micro_targets[:-1])
    return zip(micro_inputs, micro_targets)


def draw_timeline(start_iter, end_iter, server_timing, client_timing,
                  time_difference=0., save_address=None, show=False):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 4))
    timings = {
        'server': server_timing,
        'client': client_timing,
    }
    y_labels = ['client_comp', 'download', 'upload', 'server_comp']
    y_data = {
        'server_comp': [],
        'upload': [],
        'download': [],
        'client_comp': [],
    }
    total_start_time = 0
    total_end_time = 100
    for node_name, timing in timings.items():
        for i in range(start_iter, end_iter):
            for name, sub_timings in timing.break_downs[0][i].items():
                if name == 'forward':
                    color = 'green'
                    y_label = node_name + '_comp'
                elif name == 'backward':
                    color = 'blue'
                    y_label = node_name + '_comp'
                elif name == 'upload':
                    color = 'yellow'
                    y_label = name
                elif name == 'download':
                    color = 'red'
                    y_label = name
                elif name == 'update':
                    color = 'black'
                    y_label = node_name + '_comp'
                elif name == 'idle':
                    color = 'white'
                    y_label = node_name + '_comp'
                elif name == 'load':
                    color = 'orange'
                    y_label = node_name + '_comp'
                elif name == 'iter' and node_name == 'client':
                    if i == start_iter:
                        total_start_time = sub_timings[0].start_time - time_difference
                    elif i == end_iter - 1:
                        total_end_time = sub_timings[0].stop_time - time_difference
                    continue
                else:
                    continue
                y_data[y_label].append([node_name, sub_timings, color])
    for y_label in y_labels:
        for node_name, sub_timings, color in y_data[y_label]:
            for sub_timing in sub_timings:
                start_time = sub_timing.start_time - total_start_time
                if node_name == "client":
                    start_time -= time_difference
                ax.barh(y_label, sub_timing.elapsed_time, left=start_time,
                        align='center', color=color, edgecolor='black')
    ax.set_xlabel('Timeline')
    ax.set_title('Performance Analysis')
    ax.set_xlim((0, total_end_time - total_start_time))
    if save_address:
        plt.savefig(save_address)
    if show:
        plt.show()
    plt.close(fig)


def save_stats_csv(stat, file_name):
    result_dict = {
        "experiment_name": stat["experiment_name"],
        "client_index": stat["client_index"],
        "server_forward": stat["training_time"]["server"]["forward"],
        "server_backward": stat["training_time"]["server"]["backward"],
        "server_update": stat["training_time"]["server"]["update"],
        "server_idle": stat["training_time"]["server"]["idle"],
        "client_data_load": stat["training_time"]["client"]["data_load"],
        "client_forward": stat["training_time"]["client"]["forward"],
        "client_backward": stat["training_time"]["client"]["backward"],
        "client_update": stat["training_time"]["client"]["update"],
        "client_idle": stat["training_time"]["client"]["idle"],
        "upload": stat["training_time"]["upload"],
        "download": stat["training_time"]["download"],
        "total": stat["training_time"]["server"]["total"],
        "comm amount": (sum(stat["server_communication_amount"].values())
                        + sum(stat["client_communication_amount"].values())) / 1024 / 1024,
        "test_accuracy": stat["test_accuracy"],
        "train_loss": str(stat["train_loss"]),
        "train_acc": str(stat["train_acc"]),
        "val_loss": str(stat["val_loss"]),
        "val_acc": str(stat["val_acc"])
    }
    result_df = pd.DataFrame(result_dict, index=[0])
    if os.path.exists(file_name):
        result_df.to_csv(file_name, mode='a', header=False)
    else:
        result_df.to_csv(file_name)


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
