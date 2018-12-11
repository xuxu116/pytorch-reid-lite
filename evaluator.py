from __future__ import division

import argparse
import json
import sys
import os
import logging
import importlib

import torch.nn as nn

from nets.model_main import ft_net
from utils import model_utils

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, "evaluate"))


def evaluate(config):
    feature_path = config.get("feature_path", "")
    if feature_path == "" or not os.path.exists(feature_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpus"])[1:-1]
        # Load and initialize network
        net = ft_net(config,
                    model_name=config["model_params"]["model"],
                    feature_dim=config["model_params"].get("feature_dim", 256),
                    pcb_n_parts=config["model_params"].get("pcb_n_parts", 0),
                    is_training=False)
        net.train(False)

        # Set data parallel
        devices = range(len(config["gpus"]))
        if len(devices) > 1:
            net = nn.DataParallel(net)
        net = net.cuda()
        net.eval()

        config["online_net"] = net
        # Restore pretrain model
        model_utils.restore_model(config["model_path"], net)

    # Evaluate interface
    evaluate_func = None
    if config.get("benchmark_type", None):
        logging.info(
            "Using {} to evaluate the model."
            .format(config["benchmark_type"]))
        evaluate_func = importlib.import_module(
            config["benchmark_type"]).run_eval
        #config["online_net"] = net

    if evaluate_func:
        #net.eval()
        logging.info("Running evalutation: %s" %
                     config["benchmark_type"])
        evaluate_func(config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
    config = json.load(open(sys.argv[1], "r"))
    evaluate(config)
