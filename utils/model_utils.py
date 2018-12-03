import torch
import os
import logging
from collections import OrderedDict


torch_ver = torch.__version__[:3]


def save_and_evaluate(net, config, evaluate_func, save_ckpt=True,
        model_name="model.pth"):
    config["best_eval_result"] = 0.0 if "best_eval_result" not in config\
        else config["best_eval_result"]

    state_dict = net.state_dict()

    if save_ckpt:
        checkpoint_path = os.path.join(config["sub_working_dir"], model_name)
        torch.save(state_dict, checkpoint_path)
        logging.info("Model checkpoint saved to %s" % checkpoint_path)

    if evaluate_func:
        net.eval()
        logging.info("Running evalutation: %s" %
                     config["evaluation_params"]["type"])
        eval_result = evaluate_func(config)
        if eval_result > config["best_eval_result"]:
            config["best_eval_result"] = eval_result
            logging.info("New best result: {}"
                         .format(config["best_eval_result"]))
            best_checkpoint_path = os.path.join(config["sub_working_dir"],
                                                "model_best.pth")
            torch.save(state_dict, best_checkpoint_path)
            logging.info(
                "Best checkpoint saved to {}".format(best_checkpoint_path))
        else:
            logging.info("Best result: {}".format(config["best_eval_result"]))
        net.train()
        torch.cuda.empty_cache()


def restore_model(model_path, model, eval_mode=False):
    logging.info(
        "Restoring trained model from %s" % model_path
    )

    if eval_mode:
        state_dict = torch.load(
            model_path,
            map_location=lambda storage, loc: storage.cuda(0))
    else:
        state_dict = torch.load(model_path)

    if "module." in state_dict.keys()[0] and \
            "module." not in model.state_dict().keys()[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if "se_module" in k:
                name = k.replace("se_module", "se_layer")
                name = name.replace("module.", "")
                name = name.replace("se_layer", "se_module")
            else:
                name = name.replace("module.", "")
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif "module." not in state_dict.keys()[0] and \
            "module." in model.state_dict().keys()[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "module." + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    #new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    if 'classifier' in k:
    #        continue
    #    new_state_dict[k] = v
    #state_dict = new_state_dict

    # Check if there is key mismatch:
    mismatch_keys = []
    for k in model.state_dict().keys():
        if k not in state_dict:
            mismatch_keys.append(k)

    if len(mismatch_keys) > 0:
        logging.warn("[MODEL_RESTORE] number of mismatch_keys: %s"
                     % (mismatch_keys))

    model.load_state_dict(state_dict, strict=False)
