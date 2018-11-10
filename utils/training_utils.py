from __future__ import division

import time
import logging
import torch
import torch.nn as nn

from model_utils import save_and_evaluate
from nets import layers


def _get_xent_loss(config, criterion, outputs, labels, step):
    loss = criterion(outputs, labels).mean() \
        if config["model_params"].get("pcb_n_parts", 0) == 0 \
        else sum([criterion(output, labels) for output in outputs])
    return loss


def _compute_batch_acc(config, outputs, labels, step, class_balanced=False):
    batch_params = config["batch_sampling_params"]
    batch_size = config["batch_size"] \
        if not class_balanced \
        else batch_params["P"] * batch_params["K"]

    _, preds = torch.max(outputs.data, 1) \
        if config["model_params"].get("pcb_n_parts", 0) == 0 \
        else torch.max(torch.mean(torch.stack(outputs), dim=0).data, 1)
    batch_acc = torch.sum(preds == labels).item() / batch_size

    return batch_acc


def run_iter_softmax(images, labels, step, epoch, config, net, loss_dict,
                     optimizer, evaluate_func, iter_start_time,
                     io_finished_time, **kwargs):
    config["global_step"] += 1

    # Forward and backward
    optimizer.zero_grad()
    outputs = net(images, labels=labels)
    criterion = loss_dict["xent_loss"]
    loss = _get_xent_loss(config, criterion, outputs, labels, step)

    loss.backward()
    optimizer.step()

    log_step = 50 if config.get("model_parallel", False) else 50
    if step > 0 and step % log_step == 0:
        step_finished_time = time.time()
        gpu_time = float(step_finished_time - io_finished_time)
        io_time = float(io_finished_time - iter_start_time)
        example_per_second = config["batch_size"] / (gpu_time + io_time)
        io_percentage = io_time / (gpu_time + io_time)
        batch_acc = _compute_batch_acc(config, outputs, labels, step)

        logging.info(
            "epoch [%.3d] iter = %d loss = %.4f acc = %.5f example/sec = %.3f, "
            "io_percentage = %.3f" %
            (epoch, step, loss.item(), batch_acc, example_per_second,
             io_percentage)
        )

        # Write summary
        config["tensorboard_writer"].add_scalar("loss",
                                                loss.item(),
                                                config["global_step"])

        config["tensorboard_writer"].add_scalar("batch_accuray",
                                                batch_acc,
                                                config["global_step"])

    if step > 0 and step % 1000 == 0:
        save_and_evaluate(net, config, None)

    if evaluate_func and step > 0 \
            and config["evaluation_params"].get("step", None) \
            and step % config["evaluation_params"]["step"] == 0:
        save_and_evaluate(net, config, evaluate_func, save_ckpt=False)


def run_iter_triplet_loss(images, labels, config, net, loss_dict,
                          optimizer, evaluate_func, iter_start_time,
                          io_finished_time, **kwargs):
    config["global_step"] += 1
    step = config["global_step"]
    tri_loss = loss_dict["tri_loss"]

    # Forward and backward
    optimizer.zero_grad()
    if config["tri_loss_params"]["lambda_cls"]:
        # Joint training loss
        outputs = net(images, labels=labels)
        loss_tri, pull_ratio, active_triplet,\
            mean_dist_an, mean_dist_ap,\
            = tri_loss(outputs[0], labels, step)
        loss_cls = _get_xent_loss(config, loss_dict["xent_loss"], outputs[1],
                                  labels, step)
        loss = loss_cls * config["tri_loss_params"]["lambda_cls"] + \
            loss_tri * config["tri_loss_params"]["lambda_tri"]
    else:
        outputs = net(images, labels=labels)
        loss, pull_ratio, active_triplet,\
            mean_dist_an, mean_dist_ap \
            = tri_loss(outputs, labels, step)

    loss.backward()
    optimizer.step()

    if step > 0 and step % 10 == 0:
        step_finished_time = time.time()
        gpu_time = float(step_finished_time - io_finished_time)
        io_time = float(io_finished_time - iter_start_time)
        example_per_second = config["batch_size"] / (io_time + gpu_time)
        io_percentage = io_time / (gpu_time + io_time)
        if config["tri_loss_params"]["lambda_cls"]:
            logging.info(
                "global_step = %d tri_loss = %.4f xent_loss = %.4f "
                "example/sec = %.3f, io_percentage = %.3f" %
                (step, loss_tri.item(), loss_cls.item(),
                 example_per_second, io_percentage)
            )
        else:
            logging.info(
                "global_step = %d loss = %.4f example/sec = %.3f, "
                "io_percentage = %.3f" %
                (step, loss.item(),  example_per_second, io_percentage)
            )

        # Writer summaries
        config["tensorboard_writer"].add_scalar(
            "loss", loss.item(), config["global_step"])

        config["tensorboard_writer"].add_scalar(
            "AN_lt_AP_ratio", pull_ratio.item(), config["global_step"])

        config["tensorboard_writer"].add_scalar(
            "Active_Triplet", active_triplet.item(), config["global_step"])

        config["tensorboard_writer"].add_scalar(
            "Mean_Dist_Difference", mean_dist_an.item() - mean_dist_ap.item(),
            config["global_step"])

    # log triplet loss info and acc
    if step > 0 and step % 100 == 0:
        logging.info("[TRI_LOSS_INFO] AN > AP: %.2f%%; ACTIVE_TRIPLET: %d;"
                     " MEAN_DIST_AN: %.2f; MEAN_DIST_AP: %.2f" %
                     (pull_ratio.item(), active_triplet.item(),
                      mean_dist_an.item(), mean_dist_ap.item()))

        # log acc if training with joint loss
        if config["tri_loss_params"]["lambda_cls"]:
            batch_acc = _compute_batch_acc(config, outputs[1], labels, step,
                                           class_balanced=True)
            logging.info("train_accuracy: %.5f" % batch_acc)

    if step > 0 and step % 1000 == 0:
        save_and_evaluate(net, config, None)

    if evaluate_func and config["evaluation_params"].get("step", None)\
            and step % config["evaluation_params"]["step"] == 0:
        save_and_evaluate(net, config, evaluate_func, save_ckpt=False)


def get_loss_dict(config):
    use_tri_loss = config["tri_loss_params"]["margin"] > 0
    loss_dict = {}
    if use_tri_loss:
        logging.info("Using Triplet Loss: %s" % config["tri_loss_params"])
        loss_dict["tri_loss"] = layers.TripletLoss(
            config["tri_loss_params"]["margin"],
            config["tri_loss_params"]["use_adaptive_weight"]
        )

    if not use_tri_loss \
            or (use_tri_loss and config["tri_loss_params"]["lambda_cls"] > 0):
        loss_dict["xent_loss"] = nn.CrossEntropyLoss()

    return loss_dict
