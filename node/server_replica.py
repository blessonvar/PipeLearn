import logging
import torch
from util.message import Message
from util.timing import Timing
from util.util import get_optimiser, get_net
from util.profile import Profiler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServerReplica:
    def __init__(self,
                 client_index,
                 ip_address,
                 port,
                 server_socket,
                 client_socket,
                 device,
                 net,
                 criterion,
                 config):
        self.client_index = client_index
        self.ip_address = ip_address
        self.port = port
        self.server_socket = server_socket
        self.client_socket = client_socket
        self.device = device
        self.net = net
        self.complete_net = None
        self.optimiser = None
        self.criterion = criterion
        self.config = config
        self.loss = None
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.global_client_model = None
        self.global_server_model = None
        self.global_model = None
        self.test_accuracy = 0
        self.server_communication_amount = {}
        self.client_communication_amount = {}
        self.server_timing = Timing('server')
        self.client_timing = None
        self.server_profile_res = None
        self.client_profile_res = None
        logger.info("New connection added: " + ip_address + ":" + str(port))

    def update_net(self, net):
        self.net = net
        self.optimiser = get_optimiser(self.net, self.config.LAYER_NUM_ON_CLIENT, self.config.LEARNING_RATE)

    def train(self, epoch_index, aggregator, comm):
        epoch_timing = self.server_timing.break_down('epoch', epoch_index)
        epoch_timing.start()
        self.net.to(self.device)
        if self.config.LAYER_NUM_ON_CLIENT == -1:
            self.train_on_client(epoch_index, aggregator)
        elif self.config.LAYER_NUM_ON_CLIENT == 0:
            self.train_on_server(epoch_index, aggregator)
        else:
            self.train_split(epoch_index, aggregator, comm)
        epoch_timing.stop()

    def train_on_client(self, epoch_index, aggregator):
        # train
        iter_index = 0
        idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
        idle_timing.start()
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            message = Message(from_socket=self.client_socket,
                              expected_title='MSG_LOCAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
            if message.end:
                break
            idle_timing.stop()
            self.loss = message.content
            # Client model aggregation.
            if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0 and self.config.CLIENT_NUM > 1:
                self.receive_client_params(epoch_index, iter_index)
                self.global_client_model = self.aggregation(self.global_client_model, aggregator)
                self.send_client_params()
            iter_index += 1
            idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            iter_timing.stop()
        idle_timing.stop()
        message = Message(from_socket=self.client_socket,
                          expected_title='MSG_LOCAL_TRAIN_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
        train_loss, train_acc = message.content
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        # val
        if self.config.ENABLE_VAL:
            message = Message(from_socket=self.client_socket,
                              expected_title='MSG_LOCAL_VAL_LOSS_CLIENT_{}_TO_SERVER'.format(self.client_index))
            val_loss, val_acc = message.content
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

    def train_on_server(self, epoch_index, aggregator):
        # train
        self.net.train()
        iter_index = 0
        train_loss = 0.
        correct = 0
        total = 0
        idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
        idle_timing.start()
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            message = Message(
                from_socket=self.client_socket,
                expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                time_difference=self.config.TIME_DIFFERENCE
            )
            if message.end:
                break
            idle_timing.stop()
            upload_timing = self.server_timing.break_down('upload', epoch_index, iter_index)
            upload_timing.set_start_time(message.send_time)
            upload_timing.set_stop_time(message.receive_time)
            forward_timing = self.server_timing.break_down('forward', epoch_index, iter_index)
            forward_timing.start()
            activations, labels = message.content
            inputs, targets = activations.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            forward_timing.stop()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            backward_timing = self.server_timing.break_down('backward', epoch_index, iter_index)
            backward_timing.start()
            loss.backward()
            backward_timing.stop()
            self.loss = loss.item()
            update_timing = self.server_timing.break_down('update', epoch_index, iter_index)
            update_timing.start()
            self.optimiser.step()
            self.optimiser.zero_grad()
            update_timing.stop()
            # Server model aggregation.
            if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0 and self.config.CLIENT_NUM > 1:
                self.global_server_model = self.aggregation(self.net.state_dict(), aggregator)
                self.net.load_state_dict(self.global_server_model)
            iter_index += 1
            idle_timing = self.client_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            iter_timing.stop()
        idle_timing.stop()
        train_loss /= iter_index
        self.train_loss.append(train_loss)
        train_accuracy = 100. * correct / total
        self.train_acc.append(train_accuracy)
        # val
        if self.config.ENABLE_VAL:
            self.net.eval()
            iter_index = 0
            val_loss = 0.
            correct = 0
            total = 0
            with torch.no_grad():
                while True:
                    message = Message(
                        from_socket=self.client_socket,
                        expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                        time_difference=self.config.TIME_DIFFERENCE
                    )
                    if message.end:
                        break
                    activations, labels = message.content
                    inputs, targets = activations.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    iter_index += 1
            val_loss /= iter_index
            self.val_loss.append(val_loss)
            val_accuracy = 100. * correct / total
            self.val_acc.append(val_accuracy)

    def train_split(self, epoch_index, aggregator, comm):
        # train
        self.net.train()
        iter_index = 0
        train_loss = 0.
        correct = 0
        total = 0
        while True:
            iter_timing = self.server_timing.break_down('iter', epoch_index, iter_index)
            iter_timing.start()
            idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
            idle_timing.start()
            end = False
            pipe_loss = 0.
            pipe_batch_num = self.config.BATCH_SIZE // (self.config.BATCH_SIZE // self.config.PIPE_BATCH_NUM)
            for pipe_idx in range(pipe_batch_num):
                if self.config.ASYNC_COMM:
                    message = comm.receive(
                        expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index))
                else:
                    message = Message(
                        from_socket=self.client_socket,
                        expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                        time_difference=self.config.TIME_DIFFERENCE
                    )
                if message.end:
                    end = True
                    break
                upload_timing = self.server_timing.break_down('upload', epoch_index, iter_index)
                upload_timing.set_start_time(message.send_time)
                upload_timing.set_stop_time(message.receive_time)
                idle_timing.stop()
                forward_timing = self.server_timing.break_down('forward', epoch_index, iter_index)
                forward_timing.start()
                activations, labels = message.content
                inputs, targets = activations.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                if self.config.IMMEDIATE_UPDATE:
                    loss = self.criterion(outputs, targets)
                    pipe_loss = loss
                    train_loss += loss.item() / self.config.PIPE_BATCH_NUM
                else:
                    loss = self.criterion(outputs, targets) / self.config.PIPE_BATCH_NUM
                    pipe_loss += loss
                    train_loss += loss.item()
                forward_timing.stop()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                backward_timing = self.server_timing.break_down('backward', epoch_index, iter_index)
                backward_timing.start()
                loss.backward()
                backward_timing.stop()
                self.loss = loss.item()
                if self.config.IMMEDIATE_UPDATE:
                    update_timing = self.server_timing.break_down('update', epoch_index, iter_index)
                    update_timing.start()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    update_timing.stop()
                # Send gradients to client.
                idle_timing = self.server_timing.break_down('idle', epoch_index, iter_index)
                idle_timing.start()
                if self.config.ASYNC_COMM:
                    comm.send(Message(
                        title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                        content=[inputs.grad, pipe_loss],
                        sender_change=False if pipe_idx + 1 < self.config.PIPE_BATCH_NUM else True,
                        receiver_change=False if pipe_idx + 1 < self.config.PIPE_BATCH_NUM else True
                    ))
                else:
                    self.server_communication_amount["TRAINING"] = self.server_communication_amount.get(
                        "TRAINING", 0) + Message(
                        title='MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                        content=[inputs.grad, pipe_loss]
                    ).send_to(self.client_socket)
            if end:
                idle_timing.stop()
                break
            if not self.config.IMMEDIATE_UPDATE:
                update_timing = self.server_timing.break_down('update', epoch_index, iter_index)
                update_timing.start()
                self.optimiser.step()
                self.optimiser.zero_grad()
                update_timing.stop()
            # Model aggregation.
            if (iter_index + 1) % self.config.AGGREGATION_FREQ == 0 and self.config.CLIENT_NUM > 1:
                if self.complete_net is None:
                    self.complete_net = get_net(self.config, "complete")
                    self.global_model = self.complete_net.state_dict()
                self.global_server_model = self.net.state_dict()
                self.receive_client_params(epoch_index, iter_index, comm)
                self.merge_models()
                self.global_model = self.aggregation(self.global_model, aggregator)
                self.split_models()
                self.send_client_params(comm)
                self.net.load_state_dict(self.global_server_model)
            iter_index += 1
            iter_timing.stop()
        train_loss /= iter_index
        self.train_loss.append(train_loss)
        train_acc = 100. * correct / total
        self.train_acc.append(train_acc)
        # val
        if self.config.ENABLE_VAL:
            self.net.eval()
            iter_index = 0
            val_loss = 0.
            correct = 0
            total = 0
            with torch.no_grad():
                while True:
                    if self.config.ASYNC_COMM:
                        message = comm.receive(
                            expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index))
                    else:
                        message = Message(
                            from_socket=self.client_socket,
                            expected_title='MSG_LOCAL_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                            time_difference=self.config.TIME_DIFFERENCE
                        )
                    if message.end:
                        break
                    activations, labels = message.content
                    inputs, targets = activations.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    iter_index += 1
            val_loss /= iter_index
            self.val_loss.append(val_loss)
            val_acc = 100. * correct / total
            self.val_acc.append(val_acc)

    def aggregation(self, model, aggregator):
        aggregator.add_model(model)
        if self.client_index == 0:
            aggregator.aggregate()
        global_model = aggregator.get_global_model()
        aggregator.task_done()
        return global_model

    def merge_models(self):
        client_keys = list(self.global_client_model)
        server_keys = list(self.global_server_model)
        complete_keys = list(self.global_model)
        for i in range(len(client_keys)):
            self.global_model[complete_keys[i]] = self.global_client_model[client_keys[i]]
        for i in range(len(server_keys)):
            self.global_model[complete_keys[i+len(client_keys)]] = self.global_server_model[server_keys[i]]

    def split_models(self):
        client_keys = list(self.global_client_model)
        server_keys = list(self.global_server_model)
        complete_keys = list(self.global_model)
        for i in range(len(client_keys)):
            self.global_client_model[client_keys[i]] = self.global_model[complete_keys[i]]
        for i in range(len(server_keys)):
            self.global_server_model[server_keys[i]] = self.global_model[complete_keys[i+len(client_keys)]]

    def test(self):
        self.net.to(self.device)
        self.net.eval()
        if self.config.LAYER_NUM_ON_CLIENT != -1:
            while True:
                message = Message(from_socket=self.client_socket,
                                  expected_title='MSG_LOCAL_TEST_ACTIVATIONS_CLIENT_{}_TO_SERVER'.format(
                                      self.client_index))
                if message.end:
                    break
                activations, labels = message.content
                inputs, targets = activations.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                # Send gradients to client.
                self.server_communication_amount["TESTING"] = self.server_communication_amount.get(
                    "TESTING", 0) + Message(
                    title='MSG_SERVER_LOSS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                    content=[outputs, loss]
                ).send_to(self.client_socket)
        self.receive_test_accuracy()

    def profile(self):
        profiler = Profiler(self.config)
        self.server_profile_res = profiler.run()

    def send_recommend_parameters(self):
        Message(
            title='MSG_RECOMMEND_PARAMS_TO_CLIENT_{}'.format(self.client_index),
            content=[self.config.LAYER_NUM_ON_CLIENT, self.config.PIPE_BATCH_NUM]
        ).send_to(self.client_socket)

    def send_client_params(self, comm=None):
        if self.config.ASYNC_COMM:
            comm.send(Message(
                title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                content=self.global_client_model,
                sender_change=True,
                receiver_change=True,
            ))
        else:
            self.server_communication_amount["AGGREGATION"] = self.server_communication_amount.get(
                "AGGREGATION", 0) + Message(
                title='MSG_LOCAL_PARAMS_SERVER_TO_CLIENT_{}'.format(self.client_index),
                content=self.global_client_model
            ).send_to(self.client_socket)

    def send_sync_flag(self):
        Message(
            title='MSG_SYNC_FLAG_TO_CLIENT_{}'.format(self.client_index),
            content="Client {} synchronised.".format(self.client_index)
        ).send_to(self.client_socket)

    def receive_client_params(self, epoch_index, iter_index, comm=None):
        if self.config.ASYNC_COMM:
            message = comm.receive('MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.client_index))
        else:
            message = Message(
                from_socket=self.client_socket,
                expected_title='MSG_LOCAL_PARAMS_CLIENT_{}_TO_SERVER'.format(self.client_index),
                time_difference=self.config.TIME_DIFFERENCE
            )
        upload_timing = self.server_timing.break_down('upload', epoch_index, iter_index)
        upload_timing.set_start_time(message.send_time)
        upload_timing.set_stop_time(message.receive_time)
        self.global_client_model = message.content

    def receive_test_accuracy(self):
        self.test_accuracy = Message(
            from_socket=self.client_socket,
            expected_title='MSG_LOCAL_TEST_ACCURACY_CLIENT_{}_TO_SERVER'.format(self.client_index)
        ).content

    def receive_client_communication_amount(self):
        self.client_communication_amount = Message(
            from_socket=self.client_socket,
            expected_title="MSG_COMMUNICATION_AMOUNT_CLIENT_{}_TO_SERVER".format(self.client_index)
        ).content

    def receive_client_timing(self):
        self.client_timing = Message(
            from_socket=self.client_socket,
            expected_title="MSG_TIMING_CLIENT_{}_TO_SERVER".format(self.client_index)
        ).content

    def receive_client_profile_res(self):
        self.client_profile_res = Message(
            from_socket=self.client_socket,
            expected_title="MSG_PROFILE_RES_CLIENT_{}_TO_SERVER".format(self.client_index)
        ).content
