import logging
import os
import time
from copy import deepcopy
from threading import Thread

import torch
import tqdm
from torch.multiprocessing import Process, Queue

from util.aggregator import Aggregator
from util.communication import Communication
from util.profile import recommend_parameter
from util.message import Message
from util.util import draw_timeline, get_net, save_result, save_stats_csv
from node.node import Node
from node.server_replica import ServerReplica

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Server(Node):
    def __init__(self, config):
        super().__init__(config)
        self.socket.bind(("", self.config.SERVER_PORT))
        self.socket.listen()
        self.stats = []
        self.net = None
        self.client_net = None

    def run(self):
        all_processes = []
        sync = Queue(self.config.CLIENT_NUM)
        stats = Queue(self.config.CLIENT_NUM)
        aggregator = Aggregator(self.config.CLIENT_NUM)
        for _ in range(self.config.CLIENT_NUM):
            single_process = Process(target=self.run_a_process,
                                     args=(aggregator, stats, sync))
            single_process.start()
            all_processes.append(single_process)
        for single_process in all_processes:
            single_process.join()
        while not stats.empty():
            self.stats.append(stats.get())
        test_accuracy = 0
        split_points = []
        parallel_batch_nums = []
        for stat in self.stats:
            test_accuracy += stat["test_accuracy"]
            self._accumulate_communication(stat["server_communication_amount"])
            self._accumulate_communication(stat["client_communication_amount"])
            split_points.append(stat['split_point'])
            parallel_batch_nums.append(stat['parallel_batch_num'])
        self.communication_amount['TOTAL'] = sum(self.communication_amount.values())
        test_accuracy /= len(self.stats)
        training_time = self.stats[0]["training_time"]["server"]["total"]
        # Save results.
        if not os.path.exists(self.config.RESULT_DIR):
            os.mkdir(self.config.RESULT_DIR)
        with open(self.config.RESULT_TXT_ADDRESS, 'a+') as f:
            save_result(
                self.config, f, test_accuracy, training_time, self.communication_amount, split_points, parallel_batch_nums)
        logging.info("Result saved in {}".format(self.config.RESULT_TXT_ADDRESS))

    def connect(self):
        logger.info("Waiting Incoming Connections.")
        (client_socket, (ip_address, port)) = self.socket.accept()
        logger.info('Got connection from ' + str(ip_address) + ':' + str(port))
        logger.info(client_socket)
        Message(
            title="MSG_SERVER_CONFIG",
            content=self.config,
        ).send_to(client_socket)
        Message(
            title="MSG_SERVER_TIME",
            content=time.time()
        ).send_to(client_socket)
        client_config = Message(
            from_socket=client_socket, expected_title="MSG_CLIENT_CONFIG").content
        self.config.update(
            client_config,
            exceptions=["CLIENT_INDEX",
                        "BATCH_SIZE",
                        "DATA_LENGTH",
                        "LOCAL_DATA_LENGTH",
                        "TEST_DATA_LENGTH",
                        "LOCAL_TEST_DATA_LENGTH",
                        "THREAD_NUM"]
        )
        self.config.TIME_DIFFERENCE *= -1
        Message(
            title="MSG_CONNECTION_STATE",
            content="Client {} connected to the server.".format(client_config.CLIENT_INDEX)
        ).send_to(client_socket)
        return ServerReplica(
            client_config.CLIENT_INDEX,
            ip_address,
            port,
            self.socket,
            client_socket,
            self.device,
            deepcopy(self.net),
            deepcopy(self.criterion),
            self.config
        )

    def _distribute_client_model(self, client_sock):
        Message(
            title="MSG_INITIAL_MODEL",
            content=self.client_net
        ).send_to(client_sock)

    def _accumulate_communication(self, communication_amount):
        for title, amount in communication_amount.items():
            self.communication_amount[title] = self.communication_amount.get(title, 0) + amount

    def run_a_process(self, aggregator, stats, sync):
        server_replica = self.connect()
        run_profiler = False
        if server_replica.config.PROFILE_ITER_NUM > 0:
            if server_replica.config.UPLINK_BANDWIDTH > 0 and server_replica.config.DOWNLINK_BANDWIDTH > 0:
                run_profiler = True
            else:
                raise Exception("Uplink and downlink bandwidth need to be given when running profiler.")
        if self.config.LAYER_NUM_ON_CLIENT != -1 or run_profiler:
            torch.set_num_threads(max(1, self.config.THREAD_NUM // self.config.CLIENT_NUM))
        else:
            torch.set_num_threads(1)
        # profile
        if run_profiler:
            server_replica.profile()
            server_replica.receive_client_profile_res()
            split_point, batch_num = recommend_parameter(
                server_replica.client_profile_res, server_replica.server_profile_res,
                server_replica.config.UPLINK_BANDWIDTH, server_replica.config.DOWNLINK_BANDWIDTH,
                server_replica.config.LAYER_NUM, server_replica.config.BATCH_SIZE)
            server_replica.config.LAYER_NUM_ON_CLIENT = split_point
            server_replica.config.PIPE_BATCH_NUM = batch_num
            server_replica.send_recommend_parameters()
        server_net = get_net(server_replica.config, model_type="server")
        client_net = get_net(server_replica.config, model_type="client")
        server_replica.update_net(server_net)
        Message(
            title="MSG_INITIAL_MODEL",
            content=client_net
        ).send_to(server_replica.client_socket)
        # Sync the start time for all clients.
        sync.put(server_replica.config.CLIENT_INDEX)
        while not (sync.full() and sync.full() and sync.full() and sync.full()):
            time.sleep(0.1)
        server_replica.send_sync_flag()
        # Train.
        logger.info("Start training...")
        server_replica.server_timing.start()
        bar = tqdm.trange(self.config.EPOCH_NUM)
        comm = None
        comm_thread = None
        comm_thread2 = None
        if server_replica.config.ASYNC_COMM:
            comm = Communication(server_replica.client_socket,
                                 half_duplex=server_replica.config.HALF_DUPLEX,
                                 time_difference=server_replica.config.TIME_DIFFERENCE)
            comm_thread = Thread(target=comm.run, kwargs={'is_send': False})
            comm_thread.start()
            if not server_replica.config.HALF_DUPLEX:
                comm_thread2 = Thread(target=comm.run, kwargs={'is_send': True})
                comm_thread2.start()
        loss_values = []
        for epoch_index in bar:
            epoch_start_time = time.time()
            bar.set_description("Training: ")
            server_replica.train(epoch_index, aggregator, comm)
            bar.set_postfix({'training loss': "{:.2f}".format(server_replica.loss),
                             'time': "{:.2f}".format(time.time() - epoch_start_time)})
            loss_values.append(server_replica.loss)
        if server_replica.config.ASYNC_COMM:
            comm.receive(expected_title="CLOSE_COMMUNICATION_CHANNEL")
            comm_thread.join()
            if not server_replica.config.HALF_DUPLEX:
                comm.send(Message(title="CLOSE_COMMUNICATION_CHANNEL", close_comm=True))
                comm_thread2.join()
            server_replica.server_communication_amount["TRAINING"] = comm.amount
        server_replica.server_timing.stop()
        logger.info("Training done.")
        # Test.
        if self.config.ENABLE_TEST:
            logger.info("Start testing...")
            server_replica.test()
            logger.info("Testing done.")
        # Statistics.
        server_replica.receive_client_communication_amount()
        server_replica.receive_client_timing()
        # Save checkpoint.
        if self.config.MODEL_SAVE:
            if not os.path.exists(self.config.MODEL_DIR):
                os.mkdir(self.config.MODEL_DIR)
            if self.config.LAYER_NUM_ON_CLIENT != -1:
                torch.save(server_replica.global_server_model,
                           os.path.join(self.config.MODEL_DIR, self.config.MODEL_NAME + '_on_server.pth'))
        if not os.path.exists(server_replica.config.RESULT_DIR):
            os.mkdir(server_replica.config.RESULT_DIR)
        if not os.path.exists(server_replica.config.FIG_DIR):
            os.mkdir(server_replica.config.FIG_DIR)
        fig_save_address = server_replica.config.EXPERIMENT_NAME + "_client_{}".format(server_replica.client_index)
        draw_timeline(
            start_iter=0,
            end_iter=5,
            server_timing=server_replica.server_timing,
            client_timing=server_replica.client_timing,
            time_difference=server_replica.config.TIME_DIFFERENCE,
            save_address=os.path.join(server_replica.config.FIG_DIR, fig_save_address),
            show=False
        )
        stat = {
            "experiment_name": server_replica.config.EXPERIMENT_NAME,
            "client_index": server_replica.client_index,
            "training_time": {
                "server": {
                    "total": round(server_replica.server_timing.elapsed_time, 2),
                    "forward": round(server_replica.server_timing.accumulate("forward"), 2),
                    "backward": round(server_replica.server_timing.accumulate("backward"), 2),
                    "update": round(server_replica.server_timing.accumulate("update"), 2),
                    "idle": round(server_replica.server_timing.accumulate("idle"), 2),
                },
                "client": {
                    "total": round(server_replica.client_timing.elapsed_time, 2),
                    "data_load": round(server_replica.client_timing.accumulate("load"), 2),
                    "forward": round(server_replica.client_timing.accumulate("forward"), 2),
                    "backward": round(server_replica.client_timing.accumulate("backward"), 2),
                    "update": round(server_replica.client_timing.accumulate("update"), 2),
                    "idle": round(server_replica.client_timing.accumulate("idle"), 2),
                },
                "upload": round(server_replica.server_timing.accumulate("upload"), 2),
                "download": round(server_replica.client_timing.accumulate("download"), 2),
            },
            "test_accuracy": server_replica.test_accuracy,
            "server_communication_amount": server_replica.server_communication_amount,
            "client_communication_amount": server_replica.client_communication_amount,
            "train_loss": server_replica.train_loss,
            "train_acc": server_replica.train_acc,
            "val_loss": server_replica.val_loss,
            "val_acc": server_replica.val_acc,
            "split_point": server_replica.config.LAYER_NUM_ON_CLIENT,
            "parallel_batch_num": server_replica.config.PIPE_BATCH_NUM
        }
        print(stat)
        save_stats_csv(stat, server_replica.config.RESULT_CSV_ADDRESS)
        stats.put(stat)
        # Ending.
        while not (stats.full() and stats.full() and stats.full() and stats.full()):
            time.sleep(0.1)
        server_replica.send_sync_flag()
        server_replica.server_socket.close()
