import logging

from queue import Queue
from util.message import Message


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communication:
    def __init__(self, socket, half_duplex=False, time_difference=0.):
        self.socket = socket
        self.half_duplex = half_duplex
        self.time_difference = time_difference
        self.send_queue = Queue()
        self.receive_queue = Queue()
        self.amount = 0.

    def run(self, is_send=True):
        while True:
            if is_send:
                # Keep sending.
                message = self.send_queue.get()
                self.amount += message.send_to(self.socket)
                if message.close_comm:
                    logging.info("Output communication ends.")
                    break
                if self.half_duplex and message.sender_change:
                    is_send = False
            else:
                # Keep receiving.
                message = Message(from_socket=self.socket, time_difference=self.time_difference)
                self.receive_queue.put(message)
                if message.close_comm:
                    logging.info("Input communication ends.")
                    break
                if self.half_duplex and message.receiver_change:
                    is_send = True

    def send(self, message):
        self.send_queue.put(message)

    def receive(self, expected_title=None):
        message = self.receive_queue.get()
        if expected_title is not None and message.title != expected_title:
            raise Exception("Expected " + expected_title + " but received " + message.title)
        return message
