# -*- coding: utf-8 -*-
from __future__ import division

import logging
import json
import sys
import numpy as np
import cv2
import os


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
        filename = os.path.basename(path)
        splits = filename.split("_")
        label = splits[0]
        camera = splits[2]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


def _evaluate(qf, ql, qc, gf, gl, gc):
    query = qf
    score = np.dot(gf, query)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
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


class MSMTDataLoader(object):
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
            p = os.path.join(dir, target)
            images.append(p)
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
    using_onnx = config.get("model_path", "").endswith("onnx")
    if using_onnx:
        # initialize model
        logging.info("[MSMT_EVAL] Initializing trained model.")
        logging.info("Using onnx predictor.")
        from onnx_predictor import ONNXPredictor
        model = ONNXPredictor(config["model_path"], gpu=config["parallels"][0])
        input_w = model.input_size[0]
        input_h = model.input_size[1]
        batch_size = model.batch_size
    else:
        model = config["online_net"]
        input_w = config["img_w"]
        input_h = config["img_h"]
        batch_size = config["batch_size"]

    # configure data_loader
    msmt_root = "/world/data-c7/person-reid-data/MSMT17_V1/test"
    dataloaders = {
        x: MSMTDataLoader(
            os.path.join(msmt_root, x), batch_size=batch_size,
            input_w=input_w, input_h=input_h, using_onnx=using_onnx)
        for x in ["gallery", "query"]
    }
    gallery_path = dataloaders["gallery"].get_paths()
    query_path = dataloaders["query"].get_paths()

    gallery_cam, gallery_label = _get_id(gallery_path)
    query_cam, query_label = _get_id(query_path)

    # Extract feature
    logging.info("[MSMT_EVAL] Start extracting gallery feature.")
    gallery_feature = _extract_feature(model, dataloaders["gallery"],
                                       using_onnx=using_onnx)
    logging.info("[MSMT_EVAL] Start extracting query feature.")
    query_feature = _extract_feature(model, dataloaders["query"],
                                     using_onnx=using_onnx)

    # Run evaluation
    logging.info("[MSMT_EVAL] Start evaluation......")
    query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    gallery_cam = np.array(gallery_cam)
    gallery_label = np.array(gallery_label)

    CMC = np.zeros(len(gallery_label), dtype=int)
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = _evaluate(
            query_feature[i],
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam
        )

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.astype(float)
    CMC = CMC / len(query_label)  # average CMC
    logging.info("[MSMT_EVAL] top1: %f top5: %f top10: %f mAP: %f" %
                 (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config = json.load(open(sys.argv[1], "r"))
    run_eval(config)


if __name__ == "__main__":
    main()
