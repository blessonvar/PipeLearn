import logging
import time
import torch
import tqdm

from node.node import Node
from util.message import Message
from util.timing import Timing
from util.util import get_optimiser, split_batch
from util.profile import Profiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Client(Node):
    def __init__(self, client_config):
        super().__init__(client_config)
        self.port = None
        self.test_accuracy = 0
        self.global_client_model = None
        self.client_timing = Timing('client')
        self.profile_res = None

    def connect(self):
        logger.info('Connecting to Server.')
        self.socket.connect((self.config.SERVER_ADDRESS, self.config.SERVER_PORT))
        self.port = self.socket.getsockname()[1]
        server_config = Message(from_socket=self.socket, expected_title="MSG_SERVER_CONFIG").content
        self.config.update(server_config,
                           exceptions=["CLIENT_INDEX", "INDEX", "THREAD_NUM"])
        self.config.init_training_info()
        server_time = Message(from_socket=self.socket, expected_title="MSG_SERVER_TIME").content
        self.config.TIME_DIFFERENCE = server_time - time.time()
        Message(title="MSG_CLIENT_CONFIG", content=self.config).send_to(self.socket)
        connection_state = Message(from_socket=self.socket, expected_title="MSG_CONNECTION_STATE").content
        logger.info(connection_state)

    def receive_model(self):
        self.net = Message(from_socket=self.socket, expected_title="MSG_INITIAL_MODEL").content
        self.optimiser = get_optimiser(self.net, self.config.LAYER_NUM_ON_CLIENT, self.config.LEARNING_RATE)

    def train(self, data_loader, epoch_index, comm=None):
        epoch_timing = self.client_timing.break_down('epoch', epoch_index)
        epoch_timing.start()
        torch.set_num_threads(self.config.THREAD_NUM)
        self.net.to(self.device)
        if self.config.LAYER_NUM_ON_CLIENT == -1:
            self.train_on_client(data_loader, epoch_index, epoch_timing.start_time)
        elif self.config.LAYER_NUM_ON_CLIENT == 0:
            self.train_on_server(data_loader, epoch_index, epoch_timing.start_time, comm)
        else:
            self.train_split(data_loader, epoch_index, epoch_timing.start_time, comm)
        self.global_client_model = self.net.state_dict()
        epoch_timing.stop()

    def train_on_client(self, data_loader, epoch_index, start_time):
        train_data_loader, val_data_loader = data_loader
        # train
        self.net.train()
        bar = tqdm.tqdm(train_data_loader)
        train_loss = 0.
        correct = 0
        total = 0
        batch_idx = 0
        iter_timing = self.client_timing.break_down('iter', epoch_index, 0)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, 0)
        load_timing.start()
        for batch_idx, (inputs, targets) in enumerate(bar):
            load_timing.stop()
            bar.set_description("Epoch {}/{}".format(epoch_index + 1, self.config.EPOCH_NUM))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # forward pass
            forward_timing = self.client_timing.break_down('forward', epoch_index, batch_idx)
            forward_timing.start()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            forward_timing.stop()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # backward pass
            backward_timing = self.client_timing.break_down('backward', epoch_index, batch_idx)
            backward_timing.start()
            loss.backward()
            backward_timing.stop()
            Message(
                title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=loss.item()
            ).send_to(self.socket)
            # param update
            update_timing = self.client_timing.break_down('update', epoch_index, batch_idx)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # Aggregation.
            idle_timing = self.client_timing.break_down('idle', epoch_index, batch_idx)
            idle_timing.start()
            self.aggregation(epoch_index, batch_idx)
            idle_timing.stop()
            iter_timing.stop()
            bar.set_postfix({"training loss": "{:.2f}".format(loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            iter_timing = self.client_timing.break_down('iter', epoch_index, batch_idx+1)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, batch_idx+1)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        Message(
            title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            end=True
        ).send_to(self.socket)
        train_loss /= batch_idx + 1
        train_acc = 100. * correct / total
        Message(
            title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=[train_loss, train_acc]
        ).send_to(self.socket)
        # val
        if self.config.ENABLE_VAL:
            self.net.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_data_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            val_loss /= batch_idx + 1
            val_acc = 100. * correct / total
            Message(
                title='MSG_LOCAL_VAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=[val_loss, val_acc]
            ).send_to(self.socket)

    def train_on_server(self, data_loader, epoch_index, start_time, comm):
        train_data_loader, val_data_loader = data_loader
        # train
        bar = tqdm.tqdm(train_data_loader)
        current_loss = 0.
        iter_timing = self.client_timing.break_down('iter', epoch_index, 0)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, 0)
        load_timing.start()
        for batch_idx, (inputs, targets) in enumerate(bar):
            load_timing.stop()
            idle_timing = self.client_timing.break_down('idle', epoch_index, batch_idx)
            idle_timing.start()
            bar.set_description("Epoch {}/{}".format(epoch_index + 1, self.config.EPOCH_NUM))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=[inputs.cpu(), targets.cpu()]
            ).send_to(self.socket)
            # Aggregation.
            self.aggregation(epoch_index, batch_idx)
            idle_timing.stop()
            iter_timing.stop()
            bar.set_postfix({"training loss": "{:.2f}".format(current_loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            iter_timing = self.client_timing.break_down('iter', epoch_index, batch_idx+1)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, batch_idx+1)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        if self.config.ASYNC_COMM:
            comm.send(Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True))
        else:
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True
            ).send_to(self.socket)
        # val
        if self.config.ENABLE_VAL:
            for batch_idx, (inputs, targets) in enumerate(val_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                Message(
                    title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    content=[inputs.cpu(), targets.cpu()]
                ).send_to(self.socket)
            if self.config.ASYNC_COMM:
                comm.send(Message(
                    title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    end=True))
            else:
                Message(
                    title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    end=True
                ).send_to(self.socket)

    def train_split(self, data_loader, epoch_index, start_time, comm):
        train_data_loader, val_data_loader = data_loader
        # train
        self.net.train()
        bar = tqdm.tqdm(train_data_loader)
        loss = 0.
        iter_index = 0
        outputs_list = []
        # forward pass
        iter_timing = self.client_timing.break_down('iter', epoch_index, iter_index)
        iter_timing.start()
        load_timing = self.client_timing.break_down('load', epoch_index, iter_index)
        load_timing.start()
        for inputs, targets in bar:
            load_timing.stop()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            bar.set_description("Epoch {}/{}".format(epoch_index + 1, self.config.EPOCH_NUM))
            # forward pass
            for pipe_idx, (pipe_inputs, pipe_targets) in enumerate(
                    split_batch(inputs, targets, self.config.PIPE_BATCH_NUM)):
                forward_timing = self.client_timing.break_down('forward', epoch_index, iter_index)
                forward_timing.start()
                pipe_outputs = self.net(pipe_inputs)
                forward_timing.stop()
                # upload activation
                idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.config.ASYNC_COMM:
                    comm.send(Message(
                        title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                        content=[pipe_outputs.cpu(), pipe_targets.cpu()],
                        sender_change=False if pipe_idx + 1 < self.config.PIPE_BATCH_NUM else True,
                        receiver_change=False if pipe_idx + 1 < self.config.PIPE_BATCH_NUM else True))
                else:
                    self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                        title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                        content=[pipe_outputs.cpu(), pipe_targets.cpu()]
                    ).send_to(self.socket)
                idle_timing.stop()
                outputs_list.append(pipe_outputs)
            # backward pass
            for pipe_outputs in outputs_list:
                idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.config.ASYNC_COMM:
                    gradients_message = comm.receive(
                        expected_title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX))
                else:
                    gradients_message = Message(
                        from_socket=self.socket,
                        expected_title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
                        time_difference=self.config.TIME_DIFFERENCE
                    )
                download_timing = self.client_timing.break_down('download', epoch_index, iter_index)
                download_timing.set_start_time(gradients_message.send_time)
                download_timing.set_stop_time(gradients_message.receive_time)
                idle_timing.stop()
                backward_timing = self.client_timing.break_down('backward', epoch_index, iter_index)
                gradients, loss = gradients_message.content
                gradients = gradients.to(self.device)
                backward_timing.start()
                pipe_outputs.backward(gradients)
                backward_timing.stop()
            update_timing = self.client_timing.break_down('update', epoch_index, iter_index)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # Aggregation.
            idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            self.aggregation(epoch_index, iter_index, comm)
            idle_timing.stop()
            iter_index += 1
            outputs_list.clear()
            iter_timing.stop()
            bar.set_postfix({"training loss": "{:.2f}".format(loss),
                             "time": "{:.2f}".format(iter_timing.stop_time - start_time)})
            iter_timing = self.client_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            load_timing = self.client_timing.break_down('load', epoch_index, iter_index)
            load_timing.start()
        load_timing.stop()
        iter_timing.stop()
        if self.config.ASYNC_COMM:
            comm.send(Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True))
        else:
            self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                end=True
            ).send_to(self.socket)
        # val
        if self.config.ENABLE_VAL:
            self.net.eval()
            with torch.no_grad():
                for inputs, targets in val_data_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)
                    # upload activation
                    if self.config.ASYNC_COMM:
                        comm.send(Message(
                            title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                            content=[outputs.cpu(), targets.cpu()]))
                    else:
                        self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                            title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                            content=[outputs.cpu(), targets.cpu()]
                        ).send_to(self.socket)
            if self.config.ASYNC_COMM:
                comm.send(Message(
                    title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    end=True))
            else:
                self.communication_amount["TRAINING"] = self.communication_amount.get("TRAINING", 0) + Message(
                    title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    end=True
                ).send_to(self.socket)

    def aggregation(self, epoch_index, iter_idx, comm=None):
        if self.config.CLIENT_NUM > 1 and (iter_idx + 1) % self.config.AGGREGATION_FREQ == 0:
            self.global_client_model = self.net.state_dict()
            self.send_client_params(comm)
            self.receive_client_params(comm, epoch_index, iter_idx)
            self.net.load_state_dict(self.global_client_model)

    def test(self, data_loader):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.config.LAYER_NUM_ON_CLIENT == -1:
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                else:
                    outputs = inputs if self.config.LAYER_NUM_ON_CLIENT == 0 else self.net(inputs)
                    self.communication_amount["TESTING"] = self.communication_amount.get("TESTING", 0) + Message(
                        title='MSG_LOCAL_TEST_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                        content=[outputs.cpu(), targets.cpu()]
                    ).send_to(self.socket)
                    # Wait receiving server gradients.
                    gradients_message = Message(
                        from_socket=self.socket,
                        expected_title='MSG_SERVER_LOSS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX)
                    )
                    outputs, loss = gradients_message.content
                    outputs, loss = outputs.to(self.device), loss.to(self.device)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            if self.config.LAYER_NUM_ON_CLIENT != -1:
                self.communication_amount["TESTING"] = self.communication_amount.get("TESTING", 0) + Message(
                    title='MSG_LOCAL_TEST_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                    end=True
                ).send_to(self.socket)
        self.test_accuracy = 100. * correct / total
        self.send_test_accuracy()

    def profile(self):
        profiler = Profiler(self.config)
        self.profile_res = profiler.run()

    def send_client_profile_res(self):
        Message(
            title='MSG_PROFILE_RES_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=self.profile_res
        ).send_to(self.socket)

    def send_client_params(self, comm):
        if self.config.ASYNC_COMM:
            comm.send(Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=self.global_client_model,
                sender_change=True,
                receiver_change=True,
            ))
        else:
            self.communication_amount["AGGREGATION"] = self.communication_amount.get("AGGREGATION", 0) + Message(
                title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
                content=self.global_client_model
            ).send_to(self.socket)

    def receive_client_params(self, comm, epoch_index, iter_index):
        if self.config.ASYNC_COMM:
            message = comm.receive(expected_title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(
                self.config.CLIENT_INDEX))
        else:
            message = Message(
                from_socket=self.socket,
                expected_title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX),
                time_difference=self.config.TIME_DIFFERENCE
            )
        download_timing = self.client_timing.break_down('download', epoch_index, iter_index)
        download_timing.set_start_time(message.send_time)
        download_timing.set_stop_time(message.receive_time)
        self.global_client_model = message.content

    def send_test_accuracy(self):
        Message(
            title='MSG_LOCAL_TEST_ACCURACY_CLIENT_{}_TO_SERVER'.format(self.config.CLIENT_INDEX),
            content=self.test_accuracy
        ).send_to(self.socket)

    def send_client_communication_amount(self):
        Message(
            title="MSG_COMMUNICATION_AMOUNT_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
            content=self.communication_amount
        ).send_to(self.socket)

    def send_client_timing(self):
        Message(
            title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.config.CLIENT_INDEX),
            content=self.client_timing
        ).send_to(self.socket)

    def receive_sync_flag(self):
        return Message(
            from_socket=self.socket,
            expected_title='MSG_SYNC_FLAG_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX)
        ).content

    def receive_recommend_parameters(self):
        self.config.LAYER_NUM_ON_CLIENT, self.config.PIPE_BATCH_NUM = Message(
            from_socket=self.socket,
            expected_title='MSG_RECOMMEND_PARAMS_TO_CLIENT_{}'.format(self.config.CLIENT_INDEX)
        ).content
