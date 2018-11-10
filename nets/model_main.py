import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import nets_factory
import layers


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
        if pcb_n_parts > 0:
            self.Pcb = layers.Pcb(config, num_ftrs, feature_dim,
                                  pcb_n_parts, is_training=is_training,
                                  feature_mask=self.feature_mask)
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_ftrs, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(p=0.5))
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
                    self.classifier = nn.Linear(
                        config["model_params"]["feature_dim"],
                        config["num_labels"])
                    self.classifier.apply(weights_init_classifier)
                else:
                    import logging
                    logging.info("Using angle loss:%s" %
                                config["asoftmax_params"])
                    self.classifier = layers.MarginInnerProduct(config,
                            config["model_params"]["feature_dim"])


    def forward(self, x, labels=None):
        embedding = self.model(x)
        embedding = self.dropout(embedding)
        if self.pcb_n_parts > 0:
            return self.Pcb(embedding, labels)

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
        if self.losses == {"tri_loss"} or (not self.training):
            return embedding
        elif self.losses == {"xent_loss", "tri_loss"}:
            # Return both the embedding and logits if training with joint loss
            return embedding, self.classifier(embedding)
        else:
            # Otherwise return only the logits
            if self.config["asoftmax_params"]["margin"] == 0:
                return self.classifier(embedding)
            else:
                return self.classifier(embedding, labels)
