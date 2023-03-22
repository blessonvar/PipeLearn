import argparse
import config
from node.server import Server
from util.util import set_random_seed


def run(server_config=None):
    set_random_seed(1994)
    if server_config is None:
        parser = argparse.ArgumentParser(description='Server-side arguments.')
        parser.add_argument('--port', '-p', type=int, default=51000)
        parser.add_argument('--client_num', '-c', type=int, nargs='+', default=[1],
                            help='the number of clients, default [1]')
        parser.add_argument('--layer_num_on_client', '-l', type=int, default=-1,
                            help='the number of layer on clients, default -1: no partition')
        parser.add_argument('--epoch_num', '-e', type=int, default=100,
                            help='the number of training epochs, default 100')
        parser.add_argument('--same_data_size_for_each_client', '-s',
                            help='the input data size for each client is the same, default False', action='store_true')
        parser.add_argument('--aggregation_frequency', '-a', type=int, default=None,
                            help='the number of iterations after which aggregation happens')
        parser.add_argument('--batch_size', '-d', type=int, default=100, help='the size of mini-batch of data')
        parser.add_argument('--thread_num', '-t', type=int, default=0, help='the number of threads')
        parser.add_argument('--data_size', '-x', type=int, default=10000, help='the size of training data')
        parser.add_argument('--test_data_size', '-y', type=int, default=10000, help='the size of test data')
        parser.add_argument('--enable_profiler', '-z',
                            help='enable profiler, default False', action='store_true')
        args = parser.parse_args()
        server_config = config.Config(
            port=args.port,
            client_num=args.client_num,
            layer_num_on_client=args.layer_num_on_client,
            epoch_num=args.epoch_num,
            same_data_size_for_each_client=args.same_data_size_for_each_client,
            aggregation_frequency=args.aggregation_frequency,
            batch_size=args.batch_size,
            thread_num=args.thread_num,
            data_size=args.data_size,
            test_data_size=args.test_data_size,
            enable_profiler=args.enable_profiler
        )
    server = Server(server_config)
    server.run()
    return


if __name__ == '__main__':
    run()
