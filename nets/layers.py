import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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

class Pcb(nn.Module):
    def __init__(self, config, num_ftrs, feature_dim, n_parts,
                 is_training=True, **kwargs):
        super(Pcb, self).__init__()
        # args
        self.training = is_training
        self.config = config
        self.num_part = n_parts
        feature_dim = feature_dim / n_parts
        self.branch = nn.ModuleList(
            [nn.Sequential(nn.Linear(num_ftrs, feature_dim),
                           nn.BatchNorm1d(feature_dim),
                           nn.Dropout(p=0.5))
             for i in range(self.num_part)]
        )
        #self.branch.apply(weights_init_kaiming)

        if is_training:
            self.classifier = nn.ModuleList(
                [nn.Linear(feature_dim, config["num_labels"])
                 for i in range(self.num_part)]
            )

    def forward(self, x, labels):
        x_split = torch.chunk(x, self.num_part, 2)
        x_global = []   # store the concated feature
        y_split = []
        for i in range(self.num_part):
            x_temp = F.avg_pool2d(x_split[i], kernel_size=x_split[i].size()[2:])
            x_temp = x_temp.view(x_temp.size(0), -1)
            x_temp = self.branch[i](x_temp)
            if self.training:
                x_global.append(x_temp)
                y_split.append(self.classifier[i](x_temp))
            else:
                y_split.append(x_temp)

        if self.training:
            return y_split
        else:
            return torch.cat(y_split, dim=1)


class TripletLoss(nn.Module):
    def __init__(self, margin=0, use_weight=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.use_weight = use_weight
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduce=False) \
            if margin != "soft_margin" else nn.SoftMarginLoss(reduce=False)
        self.softmax = nn.Softmax(dim=1)
        self.softmin = nn.Softmin(dim=1)

    def forward(self, inputs, targets, step):
        # P x K
        n = inputs.size(0)

        # (optional) used ft_norm only on finetune situaation
        # inputs = 30 * inputs / torch.norm(inputs, p=2, dim=1, keepdim=True)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # Batch hard mining
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = targets.expand(n, n).ne(targets.expand(n, n).t())
        pos_dist = dist[pos_mask].contiguous().view(n, -1)
        neg_dist = dist[neg_mask].contiguous().view(n, -1)
        if self.use_weight:
            dist_ap = torch.sum(pos_dist.mul(self.softmax(pos_dist)),
                                1, keepdim=True).squeeze(1)
            dist_an = torch.sum(neg_dist.mul(self.softmin(neg_dist)),
                                1, keepdim=True).squeeze(1)
        else:
            dist_ap = torch.max(pos_dist, 1, keepdim=True)[0].squeeze(1)
            dist_an = torch.min(neg_dist, 1, keepdim=True)[0].squeeze(1)

        y = dist_an.data.new().resize_as_(dist_an.data).fill_(1)
        active_mask = dist_an.data.new().resize_as_(dist_an.data).fill_(1)
        y.requires_grad = False
        active_mask.requires_grad = False

        # Compute loss and statistic for logging.
        loss_mat = self.ranking_loss(dist_an, dist_ap, y) \
            if self.margin != "soft_margin"\
            else self.ranking_loss(dist_an - dist_ap, y)
        loss = loss_mat.mean()
        pull_ratio = (dist_an.data > dist_ap.data).sum().\
            float() * 100. / y.size(0)
        active_triplet = active_mask[loss_mat > 0.001].sum()
        mean_dist_an = dist_an.mean()
        mean_dist_ap = dist_ap.mean()

        return loss, pull_ratio, active_triplet, mean_dist_an, mean_dist_ap
