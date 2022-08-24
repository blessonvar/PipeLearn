import socket
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, config=None):
        self.config = config
        self.socket = socket.socket()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # Deploy the neural network.
        self.net = None
        self.optimiser = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.communication_amount = {}
