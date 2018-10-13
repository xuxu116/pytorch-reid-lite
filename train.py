from __future__ import division

import argparse
import json
import sys
import os
import logging
import time
import importlib
import subprocess
import shutil
import daemon
import glob

import torch.nn as nn
import torch.optim as optim

from input_pipeline.image_data_reader import init_data_loader
from nets.model_main import ft_net
from utils import model_utils
from utils.training_utils import run_iter_softmax
from utils.training_utils import run_iter_triplet_loss
from utils.training_utils import get_loss_dict

from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, "evaluate"))


def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.model.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["fine_tune"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["fc_lr"]},
            {"params": net.model.parameters(), "lr": config["lr"]["base_lr"]},
        ]
    else:
        for p in net.model.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["fc_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["weight_decay"])
    elif config["optimizer"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, momentum=0.9,
                                  weight_decay=config["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["weight_decay"],
                              nesterov=(config["optimizer"] == "nesterov"))
    return optimizer


def _run_train_loop(data_loader, config, net,
                    loss_dict, optimizer, evaluate_func,
                    lr_scheduler):
    use_tri_loss = "tri_loss" in loss_dict
    if use_tri_loss or config["batch_sampling_params"]["class_balanced"]:
        assert config["lr"]["decay_step"] > 2000, \
            "lr decay_step is too small for class balance sampling."

        # Tri loss params
        tri_epoch = 1
        iter_fun = run_iter_triplet_loss if use_tri_loss else run_iter_softmax

        while True:
            data_loader_iter = iter(data_loader)
            for step in range(len(data_loader)):
                iter_start_time = time.time()
                images, labels = data_loader_iter.next()
                io_finished_time = time.time()
                lr_scheduler.step()
                iter_fun(
                    images=images.cuda(),
                    labels=labels.cuda(),
                    config=config,
                    net=net,
                    loss_dict=loss_dict,
                    optimizer=optimizer,
                    evaluate_func=evaluate_func,
                    step=config["global_step"],
                    epoch=0,
                    iter_start_time=iter_start_time,
                    io_finished_time=io_finished_time
                )
            tri_epoch += 1
    else:
        for epoch in range(config["epochs"]):
            lr_scheduler.step()
            data_loader_iter = iter(data_loader)
            for step in range(len(data_loader)):
                iter_start_time = time.time()
                images, labels = data_loader_iter.next()
                io_finished_time = time.time()
                run_iter_softmax(
                    images=images.cuda(),
                    labels=labels.cuda(),
                    step=step,
                    epoch=epoch,
                    config=config,
                    net=net,
                    loss_dict=loss_dict,
                    optimizer=optimizer,
                    evaluate_func=evaluate_func,
                    iter_start_time=iter_start_time,
                    io_finished_time=io_finished_time
                )
            # evaluate the model after each epoch
            model_utils.save_and_evaluate(net, config, evaluate_func)


def train(config, data_loader):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["parallels"])[1:-1]
    config["global_step"] = 0

    # Get all losses
    loss_dict = get_loss_dict(config)

    # Load and initialize network
    net = ft_net(config,
                 model_name=config["model_params"]["model"],
                 feature_dim=config["model_params"].get("feature_dim", 256),
                 pcb_n_parts=config["model_params"].get("pcb_n_parts", 0),
                 loss_dict=loss_dict)
    net.train(True)

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    devices = range(len(config["parallels"]))
    if len(devices) > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if os.path.exists(config.get("pretrain_snapshot", "")):
        model_utils.restore_model(config["pretrain_snapshot"], net)

    # Evaluate interface
    evaluate_func = None
    if config["evaluation_params"].get("type", None):
        logging.info(
            "Using {} to evaluate the model."
            .format(config["evaluation_params"]["type"]))
        evaluate_func = importlib.import_module(
            config["evaluation_params"]["type"]).run_eval
        config["online_net"] = net

    # Start the training loop
    logging.info("Kicking off training.")
    _run_train_loop(
        data_loader=data_loader, lr_scheduler=lr_scheduler,
        config=config, net=net, loss_dict=loss_dict, optimizer=optimizer,
        evaluate_func=evaluate_func
    )


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s %(filename)s] %(message)s")

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--operation", default="", type=str)
    parser.add_argument("--config_path", default="./params.json", type=str)
    parser.add_argument("--sub_working_dir", default="", type=str)
    args = parser.parse_args()
    config = json.load(open(args.config_path, "r"))

    if args.operation == "start_train":
        # Create sub working dir
        dirname = args.sub_working_dir \
            if args.sub_working_dir != "" else str(time.time())
        sub_working_dir = os.path.join(config["working_dir"], dirname)
        if not os.path.exists(sub_working_dir):
            os.mkdir(sub_working_dir)
        else:
            # remove tensorboard log file
            for f in glob.glob(os.path.join(dirname, "event*")):
                os.remove(f)

        # Configure log file and place json to sub working dir
        log_file = open(os.path.join(sub_working_dir, "train_log"), "w", 1)
        shutil.copy(args.config_path, sub_working_dir)

        # Start training as a subprocess
        call_args = "python train.py --config_path %s --sub_working_dir %s" % \
            (args.config_path, sub_working_dir)
        subprocess.Popen(
            call_args.split(),
            # cwd=os.path.expanduser("~/pytorch-reid"),
            stdout=log_file,
            stderr=log_file)
        logging.info(
            "Train process has been started, please refer to %s for details." %
            os.path.join(sub_working_dir, "train_log")
        )
    else:
        config["batch_size"] = config["batch_sampling_params"]["batch_size"] * \
            len(config["parallels"])
        data_loader = init_data_loader(
            config,
            num_processes=config.get("num_preprocess_workers",
                                     len(config["parallels"]) * 2))
        config["sub_working_dir"] = args.sub_working_dir

        # Creat tf_summary writer
        config["tensorboard_writer"] = SummaryWriter(args.sub_working_dir)

        # Start training
        train(config, data_loader)

if __name__ == "__main__":
    ctx = daemon.DaemonContext(
        working_directory=os.path.dirname(os.path.abspath(__file__)),
        stdout=sys.__stdout__,
        stderr=sys.__stdout__,
        prevent_core=False)

    with ctx:
        main()
