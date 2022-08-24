import argparse
import gc
import logging
import os
from threading import Thread

import torch

import config
from node.client import Client
from util.communication import Communication
from util.message import Message
from util.util import get_data_loader, set_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(client_config=None):
    set_random_seed(1994)
    if client_config is None:
        parser = argparse.ArgumentParser(description='Client-side arguments.')
        parser.add_argument('--client_index', '-i', type=int, default=0, help='the index of client, required')
        parser.add_argument('--port', '-p', type=int, default=51000)
        args = parser.parse_args()
        client_config = config.Config(
            client_index=args.client_index,
            port=args.port,
        )
    local_test_accuracy = run_a_client(client_config)
    logger.info('Local Test Accuracy: {}'.format(local_test_accuracy))


def run_a_client(client_config):
    client = Client(client_config)
    client.connect()
    # set bandwidth
    bandwidth_reset = False
    if client_config.UPLINK_BANDWIDTH > 0 and client_config.DOWNLINK_BANDWIDTH > 0:
        os.system('sudo wondershaper -c -a eth0')
        os.system('sudo wondershaper -a eth0 -u {} -d {}'.format(
            client_config.UPLINK_BANDWIDTH * 1024, client_config.DOWNLINK_BANDWIDTH * 1024))
        bandwidth_reset = True
    # profile
    if (client.config.PROFILE_ITER_NUM > 0 and client.config.UPLINK_BANDWIDTH > 0
            and client.config.DOWNLINK_BANDWIDTH > 0):
        client.profile()
        client.send_client_profile_res()
        client.receive_recommend_parameters()
    client.receive_model()
    client.receive_sync_flag()
    # Train.
    client.client_timing.start()
    # Async communication setup.
    comm = None
    comm_thread = None
    comm_thread2 = None
    if client.config.ASYNC_COMM:
        comm = Communication(client.socket,
                             half_duplex=client.config.HALF_DUPLEX,
                             time_difference=client.config.TIME_DIFFERENCE)
        comm_thread = Thread(target=comm.run, kwargs={'is_send': True})
        comm_thread.start()
        if not client.config.HALF_DUPLEX:
            comm_thread2 = Thread(target=comm.run, kwargs={'is_send': False})
            comm_thread2.start()
    # Train by epochs.
    for epoch_index in range(client.config.EPOCH_NUM):
        train_data_loader = get_data_loader(client.config, set_name='train')
        val_data_loader = get_data_loader(client.config, set_name='val')
        client.train([train_data_loader, val_data_loader], epoch_index, comm)
    if client.config.ASYNC_COMM:
        comm.send(Message(title="CLOSE_COMMUNICATION_CHANNEL", close_comm=True))
        comm_thread.join()
        if not client.config.HALF_DUPLEX:
            comm.receive(expected_title="CLOSE_COMMUNICATION_CHANNEL")
            comm_thread2.join()
        client.communication_amount['TRAINING'] = comm.amount
    client.client_timing.stop()
    # Test.
    if client.config.ENABLE_TEST:
        test_data_loader = get_data_loader(client.config, set_name='test')
        client.test(test_data_loader)
    # Statistics.
    client.send_client_communication_amount()
    client.send_client_timing()
    # Save checkpoint.
    if client.config.MODEL_SAVE:
        if not os.path.exists(client.config.MODEL_DIR):
            os.mkdir(client.config.MODEL_DIR)
        torch.save(client.global_client_model,
                   os.path.join(client.config.MODEL_DIR,
                                client.config.MODEL_NAME + '_on_client{}.pth'.format(client.config.CLIENT_INDEX)))
    client.receive_sync_flag()
    client.socket.close()
    if bandwidth_reset:
        os.system('sudo wondershaper -c -a eth0')
    test_accuracy = client.test_accuracy
    del client
    gc.collect()
    return test_accuracy


if __name__ == '__main__':
    run()
