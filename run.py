import argparse
import json
import logging

from config import Config
import run_client
import run_server

parser = argparse.ArgumentParser(description='Arguments for running multiple experiments.')
parser.add_argument('--on_server', '-s',
                    help='split mini-batch across replicas in the same client, default False', action='store_true')
parser.add_argument('--client_index', '-i', type=int, default=-1,
                    help='index of the client, default -1')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    with open("config.json", "r") as f:
        config_s = json.load(f)
    logger.info("Experiment: {}".format(config_s["experiment_name"]))
    this_config = Config(**config_s)
    if args.on_server:
        logger.info("Running on server.")
        run_func = run_server.run
    else:
        logger.info("Running on client.")
        this_config.CLIENT_INDEX = args.client_index
        run_func = run_client.run
    run_func(this_config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
