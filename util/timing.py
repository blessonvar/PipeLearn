import time


class Timing:
    def __init__(self, name, epoch_index=-1, iter_index=-1):
        self.name = name
        self.epoch_no = epoch_index
        self.iter_no = iter_index
        self.start_time = None
        self.stop_time = None
        self.elapsed_time = 0
        self.break_downs = {}

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        if self.start_time:
            self.elapsed_time = self.stop_time - self.start_time

    def set_start_time(self, start_time):
        self.start_time = start_time
        if self.stop_time:
            self.elapsed_time = self.stop_time - self.start_time

    def set_stop_time(self, stop_time):
        self.stop_time = stop_time
        if self.start_time:
            self.elapsed_time = self.stop_time - self.start_time

    def break_down(self, name, epoch_index=-1, iter_index=-1):
        sub_timing = Timing(name, epoch_index, iter_index)
        if epoch_index in self.break_downs.keys():
            if iter_index in self.break_downs[epoch_index].keys():
                if name in self.break_downs[epoch_index][iter_index]:
                    self.break_downs[epoch_index][iter_index][name].append(sub_timing)
                else:
                    self.break_downs[epoch_index][iter_index][name] = [sub_timing]
            else:
                self.break_downs[epoch_index][iter_index] = {name: [sub_timing]}
        else:
            self.break_downs[epoch_index] = {iter_index: {name: [sub_timing]}}
        return sub_timing

    def accumulate(self, _name=None, _epoch_index=None, _iter_index=None):
        accumulate_time = 0.
        for epoch_index, epoch_timings in self.break_downs.items():
            for iter_index, iter_timings in epoch_timings.items():
                for name, sub_timings in iter_timings.items():
                    if ((_epoch_index is None or _epoch_index == epoch_index)
                            and (_iter_index is None or _iter_index == iter_index)
                            and (_name is None or _name == name)):
                        for sub_timing in sub_timings:
                            accumulate_time += sub_timing.elapsed_time
        return accumulate_time
