import logging
import pickle
import socket
import struct
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Message:
    def __init__(self,
                 title=None,
                 content=None,
                 end=False,
                 sender_change=False,
                 receiver_change=False,
                 close_comm=False,
                 from_socket=None,
                 expected_title=None,
                 time_difference=0.):
        self.title = title
        self.content = content
        self.end = end
        self.sender_change = sender_change
        self.receiver_change = receiver_change
        self.close_comm = close_comm
        self.send_time = None
        self.receive_time = None
        self.elapsed_time = None
        if from_socket:
            self.receive_from(from_socket, expected_title, time_difference)

    def send_to(self, sock):
        self.send_time = time.time()
        message = pickle.dumps([self.title, self.content, self.end,
                                self.sender_change, self.receiver_change, self.close_comm, self.send_time])
        sock.sendall(struct.pack(">I", len(message)))
        sock.sendall(message)
        logger.debug(str(self.title) + 'sent to' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
        return len(message)

    def receive_from(self, sock, expected_title=None, time_difference=0.):
        message_len = struct.unpack(">I", sock.recv(4, socket.MSG_WAITALL))[0]
        message = pickle.loads(sock.recv(message_len, socket.MSG_WAITALL))
        self.title, self.content, self.end, self.sender_change, self.receiver_change, self.close_comm, self.send_time \
            = message
        self.send_time -= time_difference
        self.receive_time = time.time()
        self.elapsed_time = self.receive_time - self.send_time
        logger.debug(str(self.title) + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
        if expected_title is not None and self.title != expected_title:
            raise Exception("Expected " + expected_title + " but received " + self.title)

    def __str__(self):
        return "[Title] " + str(self.title) + " [Content] " + str(self.content) + " [End] " + str(self.end)
