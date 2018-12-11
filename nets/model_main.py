import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import nets_factory
import layers

from layers import feature_erasing
#from batchnorm import BatchNorm2d
#from batchnorm import BatchNorm1d


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "MarginInnerProduct" in classname:
        init.normal_(m.weight.data, std=0.001)
        if hasattr(m, "bias") and (m.bias is not None):
            init.constant_(m.bias.data, 0.0)


class ft_net(nn.Module):
    def __init__(self, config, model_name="", feature_dim=256,
                 pcb_n_parts=0, is_training=True, loss_dict={}):
        super(ft_net, self).__init__()
        self.config = config
        self.pcb_n_parts = pcb_n_parts
        self.training = is_training
        self.losses = {l for l in loss_dict.keys()}
        self.dropout = nn.Dropout(p=0.5)
        self.feature_mask=config["model_params"].get("feature_mask", False)

        # Build base network
        network_fn = nets_factory.network_fn[model_name]
        model_ft = network_fn(
            pretrained=config.get("imagenet_pretrain", False),
            last_conv_stride=config["model_params"].get("last_conv_stride", 1)
        )
        self.model = model_ft  # stem network

        # Embedding layer
        num_ftrs = model_ft.num_ftrs
        self.feature_g_dim = config["model_params"].get("feature_gobal_dim", 0)
        feature_g_dim = self.feature_g_dim
        if self.feature_g_dim > 0:
            self.fc_g = nn.Sequential(
                nn.Linear(num_ftrs, feature_g_dim),
                nn.BatchNorm1d(feature_g_dim),
                nn.Dropout(p=0.5)
                )
            self.classifier_g = nn.Linear(
                feature_g_dim,
                config["num_labels"])
            self.classifier_g.apply(weights_init_classifier)
        if self.feature_mask:
            self.mask_g = nn.Sequential(
                nn.Linear(num_ftrs, feature_g_dim),
                nn.BatchNorm1d(feature_g_dim),
                nn.Dropout(p=0.5),
                nn.Sigmoid())

        if pcb_n_parts > 0:
            self.Pcb = layers.Pcb(config, num_ftrs, feature_dim,
                                  pcb_n_parts, is_training=is_training,
                                  feature_mask=self.feature_mask)
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_ftrs, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(p=0.5)
                )
            #self.fc = nn.Linear(num_ftrs, feature_dim)
            if self.feature_mask:
                self.mask = nn.Sequential(
                    nn.Linear(num_ftrs, feature_dim),
                    nn.BatchNorm1d(feature_dim),
                    nn.Dropout(p=0.5),
                    nn.Sigmoid())

            self.fc.apply(weights_init_kaiming)

        # Classification layer
        if self.training and pcb_n_parts == 0:
            self.classifier = None
            if "xent_loss" in self.losses or True:
                if config["asoftmax_params"]["margin"] == 0:
                    coef = 2 if config["gan_params"]["input_dim"] > 0 and \
                            config["gan_params"]["adv_train"] else 1
                    self.classifier = nn.Linear(
                        config["model_params"]["feature_dim"],
                        config["num_labels"] * coef)
                    self.classifier.apply(weights_init_classifier)
                else:
                    import logging
                    logging.info("Using angle loss:%s" %
                                config["asoftmax_params"])
                    self.classifier = layers.MarginInnerProduct(config,
                            config["model_params"]["feature_dim"])


    def forward(self, x, labels=None, return_feature=False):
        embedding = self.model(x)

        if self.feature_g_dim > 0:
            #embedding_g = self.dropout(embedding_g)
            embedding_g = feature_erasing(embedding, 0, self.training)
            embedding_g = F.adaptive_max_pool2d(embedding_g, (1, 1))
            embedding_g = embedding_g.view(embedding_g.size(0), -1)
            embedding_g = self.fc_g(embedding_g)
            if self.feature_mask and False:
                mask_g = F.adaptive_avg_pool2d(embedding_gr, (1, 1))
                mask_g = mask_g.view(mask_g.size(0), -1)
                mask_g = self.mask_g(mask_g)
                embedding_g = embedding_g * mask_g


        #embedding = self.dropout(embedding)
        if self.pcb_n_parts > 0:
            if self.training:
                return self.Pcb(embedding, labels, return_feature=return_feature),\
                        self.classifier_g(embedding_g), embedding_g
            else:
                return torch.cat([self.Pcb(embedding, labels,
                        return_feature=return_feature), embedding_g], dim=1)


        # Global average pool
        embedding = F.avg_pool2d(embedding, kernel_size=embedding.size()[2:])

        # embedding layer
        embedding_r = embedding.view(embedding.size(0), -1)
        embedding = self.fc(embedding_r)
        #embedding = embedding_r
        if self.feature_mask:
            mask = self.mask(embedding_r)
            embedding = embedding * mask

        # Return only the embedding in model deploy
        if self.losses == {"tri_loss"} or (not self.training) or labels is None:
            return embedding
        else:
            if self.config["model_params"]["spectral_trans"] > 0:
                logits = [self.classifier(embedding),
                        self.classifier(spectrual_transform(embedding,
                                                            self.config["model_params"]["spectral_trans"],
                                                            self.config))]
            else:
                logits = [self.classifier(embedding)]

            if self.losses == {"xent_loss", "tri_loss"}:
                # Return both the embedding and logits if training with joint loss
                return embedding, logits
            else:
                # Otherwise return only the logits
                if self.config["asoftmax_params"]["margin"] == 0:
                    if return_feature:
                        return embedding, logits
                    else:
                        return logits
                else:
                    if return_feature:
                        return embedding, logits
                    else:
                        return logits


def spectrual_transform(feature, temp, config):
    # feature shape: 32 * 256
    if config["acc"] < 0.9:
        config["st_mean"] = 0
        return feature
    feature_norm = feature.pow(2).sum(1).pow(0.5)
    feature_n = feature / feature_norm.view(-1, 1)
    w = torch.mm(feature_n, torch.transpose(feature_n, 0 ,1)) / temp
    config["affinity"] = w.cpu().data.numpy()
    #w = torch.exp(w)
    w = w - torch.max(w)
    dist = F.softmax(w, 0)
    #dist = torch.ones(32).cuda()
    #dist = torch.diag(dist)
    dia_dist = torch.diag(dist, 0)
    config["st_mean"] = torch.mean(dia_dist)
    return torch.mm(dist, feature)
