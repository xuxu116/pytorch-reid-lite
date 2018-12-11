# -*- coding: utf-8 -*-
from __future__ import division

import logging
import json
import sys
import numpy as np
import cv2
import os
import pickle
import time


def _extract_feature(model, dataloaders, using_onnx):
    features = None
    for images in dataloaders.get_batch():
        if using_onnx:
            outputs = model.predict(images)
            f = np.asarray(outputs)
        else:
            import torch
            images = torch.from_numpy(images)
            images = images.cuda()
            with torch.no_grad():
                f = model(images)
                f = f.cpu().numpy()
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        if features is None:
            features = f
        else:
            features = np.concatenate((features, f), axis=0)
    return features


def _get_id(img_path):
    camera_id = []
    labels = []
    for path in img_path:
        filename = path.split("/")[-1]
        label = filename[0:4]
        camera = filename.split("c")[1]
        if label[0:2] == "-1":
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def _softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
        return e_x / e_x.sum(axis=1).reshape(-1, 1)


def _feature_refine(feature, temper):
    w = np.dot(feature, np.transpose(feature)) / temper
    #w = np.exp(w)
    w = _softmax(w)
    return np.dot(w, feature)

def _evaluate(qf, ql, qc, gf, gl, gc, temper=0):
    '''
    qf: query feature
    ql: query label
    qc: query camera
    '''
    query = qf
    score = np.dot(gf, query)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    if temper > 0:
        feature_to_refine = gf[index[:50], :]
        feature_to_refine = _feature_refine(feature_to_refine, temper)
        score = np.dot(feature_to_refine, query)
        index_r = np.argsort(score)
        index_r = index_r[::-1]
        index[:50] = index[:50][index_r]

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = _compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def _compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index), dtype=int)
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere((mask == True))
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


class MarketDataLoader(object):
    def __init__(self, src_folder, batch_size, input_w, input_h,
                 using_onnx=False):
        self.samples = self.make_dataset(src_folder)
        self.batch_size = batch_size
        self.input_w = input_w
        self.input_h = input_h
        self.using_onnx = using_onnx

    def make_dataset(self, dir):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def get_paths(self):
        return self.samples

    def get_batch(self):
        for i in range(0, len(self.samples), self.batch_size):
            images = []
            if not self.using_onnx:
                for path in self.samples[i: i+self.batch_size]:
                    image = cv2.imread(path)
                    image = cv2.resize(image, (self.input_w, self.input_h))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                images = np.asarray(images, dtype=np.float32)
                images = ((images / 255.0) - [0.485, 0.456, 0.406]) \
                    / [0.229, 0.224, 0.225]
                images = np.transpose(images, (0, 3, 1, 2))
                images = images.astype(np.float32)
                yield images
            else:
                for path in self.samples[i: i+self.batch_size]:
                    image = cv2.imread(path)
                    images.append(image)
                yield images


def run_eval(config):
    feature_path = config.get("feature_path", "")
    using_onnx = config.get("model_path", "").endswith("onnx")
    if using_onnx:
        # initialize model
        logging.info("[MARKET_EVAL] Initializing trained model.")
        logging.info("Using onnx predictor.")
        from onnx_predictor import ONNXPredictor
        model = ONNXPredictor(config["model_path"], gpu=config["parallels"][0])
        input_w = model.input_size[0]
        input_h = model.input_size[1]
        batch_size = model.batch_size
    else:
        if feature_path == "" or not os.path.exists(feature_path):
            model = config["online_net"]
        if config.has_key("data_augmentation"):
            aug_params = config["data_augmentation"]
            if aug_params.get("crop_h", 0) > 0 and aug_params.get("crop_w", 0) > 0:
                input_w = aug_params["crop_w"]
                input_h = aug_params["crop_h"]
        else:
            input_w = config["img_w"]
            input_h = config["img_h"]
        batch_size = config["batch_size"]

    # configure data_loader
    market1501_origin = "/world/data-gpu-94/vincentfung13/market_pytorch_eval"
    dataloaders = {
        x: MarketDataLoader(
            os.path.join(market1501_origin, x), batch_size=batch_size,
            input_w=input_w, input_h=input_h, using_onnx=using_onnx)
        for x in ["gallery", "query"]
    }
    gallery_path = dataloaders["gallery"].get_paths()
    query_path = dataloaders["query"].get_paths()

    gallery_cam, gallery_label = _get_id(gallery_path)
    query_cam, query_label = _get_id(query_path)

    if feature_path != "" and os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            gallery_feature, query_feature = pickle.load(f)
    else:
        # Extract feature
        logging.info("[MARKET_EVAL] Start extracting gallery feature.")
        gallery_feature = _extract_feature(model, dataloaders["gallery"],
                                        using_onnx=using_onnx)
        logging.info("[MARKET_EVAL] Start extracting query feature.")
        query_feature = _extract_feature(model, dataloaders["query"],
                                        using_onnx=using_onnx)
        if feature_path != "":
            with open(feature_path, 'w') as f:
                pickle.dump((gallery_feature, query_feature), f)
            logging.info("[MARKET_EVAL] Feature saved")

    # Run evaluation
    logging.info("[MARKET_EVAL] Start evaluation......")
    query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    gallery_cam = np.array(gallery_cam)
    gallery_label = np.array(gallery_label)

    CMC = np.zeros(len(gallery_label), dtype=int)
    ap = 0.0
    temper = config.get("spectral_transform", 0)

    start = time.time()
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = _evaluate(
            query_feature[i],
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
            temper
        )

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    end = time.time()

    CMC = CMC.astype(float)
    CMC = CMC / len(query_label)  # average CMC
    logging.info("[MARKET_EVAL] top1: %f top5: %f top10: %f mAP: %f; %.2fs" %
                 (CMC[0], CMC[4], CMC[9], ap / len(query_label), end-start))
    return CMC[0]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config = json.load(open(sys.argv[1], "r"))
    run_eval(config)


if __name__ == "__main__":
    main()
