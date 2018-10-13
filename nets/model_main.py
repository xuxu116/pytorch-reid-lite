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

        # Build base network
        network_fn = nets_factory.network_fn[model_name]
        model_ft = network_fn(
            pretrained=config.get("imagenet_pretrain", False)
        )
        self.model = model_ft  # stem network

        # Embedding layer
        num_ftrs = model_ft.num_ftrs
        if pcb_n_parts > 0:
            self.Pcb = layers.Pcb(config, num_ftrs, feature_dim,
                                  pcb_n_parts, is_training=is_training)
        else:
            self.fc = nn.Linear(num_ftrs, feature_dim)
            self.fc.apply(weights_init_kaiming)

        # Classification layer
        if self.training and pcb_n_parts == 0:
            self.classifier = None
            if "xent_loss" in self.losses:
                self.classifier = nn.Linear(
                    config["model_params"]["feature_dim"],
                    config["num_labels"])
                self.classifier.apply(weights_init_classifier)

    def forward(self, x, labels=None):
        embedding = self.model(x)
        if self.pcb_n_parts > 0:
            return self.Pcb(embedding, labels)

        # Global average pool
        embedding = F.avg_pool2d(embedding, kernel_size=embedding.size()[2:])

        # embedding layer
        embedding = embedding.view(embedding.size(0), -1)
        embedding = self.fc(embedding)

        # Return only the embedding in model deploy
        if self.losses == {"tri_loss"} or (not self.training):
            return embedding
        elif self.losses == {"xent_loss", "tri_loss"}:
            # Return both the embedding and logits if training with joint loss
            return embedding, self.classifier(embedding)
        else:
            # Otherwise return only the logits
            return self.classifier(embedding)
