from torch.multiprocessing import Queue, Event
from util.util import get_average_model


class Aggregator:
    def __init__(self, model_num):
        self.model_num = model_num
        self.models = Queue(model_num)
        self.weights = Queue(model_num)
        self.global_models = Queue(model_num)
        self.event = Event()

    def add_model(self, model, weight=None):
        self.models.put(model, block=False)
        self.weights.put(weight, block=False)

    def add_global_model(self, model):
        for _ in range(self.model_num):
            self.global_models.put(model)
        self.event.clear()

    def aggregate(self, timing=None):
        models = []
        weights = []
        for _ in range(self.model_num):
            models.append(self.models.get())
            weight = self.weights.get()
            if weight is not None:
                weights.append(weight)
        if timing:
            timing.start()
        global_model = get_average_model(models, weights)
        if timing:
            timing.stop()
        self.add_global_model(global_model)

    def get_global_model(self):
        global_model = self.global_models.get()
        if self.global_models.empty():
            self.event.set()
        return global_model

    def task_done(self):
        self.event.wait()
