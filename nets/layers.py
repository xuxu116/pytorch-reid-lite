import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Parameter


class MarginInnerProduct(nn.Module):
    def __init__(self, config, in_units, out_units=-1, **kwargs):
        super(MarginInnerProduct, self).__init__()
        #  args
        self.in_units = in_units
        self.out_units = config["num_labels"] if out_units == -1 else out_units
        if config["gan_params"].get("adv_train", False):
            self.out_units *= 2
        self.config = config

        #  margin type
        self.margin = config["asoftmax_params"].get("margin", [0.35])
        self.s = config["asoftmax_params"].get("scale", 30.0)

        #  training parameter
        self.weight = Parameter(torch.Tensor(self.out_units, self.in_units),
                                requires_grad=True)
        self.unlabel_weight = None
        """
        if config["asoftmax_params"].get("unlabel_fold", 0) > 0:
            self.unlabel_bs = config["gan_params"]["batch_size"]
            self.unlabel_size = config["asoftmax_params"]["unlabel_fold"] * self.unlabel_bs
            self.unlabel_weight = Parameter(torch.Tensor(self.in_units,
                                            self.unlabel_size),
                                            requires_grad=True)
            self.unlabel_weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
            self.current_fold = 0
        """
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.step = config["asoftmax_params"].get("step", 30000)

    def forward(self, x, labels):
        w = torch.tanspose(self.weight, 0, 1)

        x_norm = x.pow(2).sum(1).pow(0.5)
        w_norm = w.pow(2).sum(0).pow(0.5)
        x_mean = x_norm.mean()
        w_mean = w_norm.mean()
        #self.weight.data = w / w_norm 
        
        """
        if update_unlabel:
            unlabel_f = torch.transpose(x, 0, 1)
            f_norm = unlabel_f.pow(2).sum(0).pow(0.5)
            start = self.current_fold * self.unlabel_bs
            end = (self.current_fold + 1) * self.unlabel_bs
            self.unlabel_weight.data[:, start: end] = unlabel_f / f_norm
            self.current_fold = self.current_fold + 1 
            if self.current_fold == self.config["asoftmax_params"]["unlabel_fold"]:
                    self.current_fold = 0
            return 
        if self.unlabel_weight is not None:
            logit_un = x.mm(self.unlabel_weight)
        """

        # compute cosine theta
        cos_theta = x.mm(w) / x_norm.view(-1, 1) / w_norm.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        current_idx = min(int(self.config["global_step"] / self.step),
            len(self.margin) - 1)
        current_margin = self.margin[current_idx]
        if self.s == 0:
            cos_theta_margin = ((1 + current_margin) * cos_theta - current_margin) * x_norm.view(-1,1)
        else:
            cos_theta_margin = cos_theta - current_margin

        # get ground truth indices
        target = labels.view(-1, 1)  # size = (B, 1)
        index = cos_theta.data * 0.0  # size = (B, Classnum)
        index.requires_grad = False

        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        if self.s == 0:
            output = x.mm(w) / w_norm
            output[index] -= output[index]
            output[index] += cos_theta_margin[index]
        else: 
            output = cos_theta * 1.0
            output[index] -= cos_theta[index]
            output[index] += cos_theta_margin[index]
            output *= self.s  # scale up in order to make softmax work
        if self.unlabel_weight is not None:
            return torch.cat([output, logit_un], 1)
        else:
            return output


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
                 is_training=True, feature_mask=False, **kwargs):
        super(Pcb, self).__init__()
        # args
        self.training = is_training
        self.feature_mask = feature_mask
        self.config = config
        self.num_part = n_parts
        feature_dim = feature_dim / n_parts
        self.branch = nn.ModuleList(
            [nn.Sequential(nn.Linear(num_ftrs, feature_dim),
                           nn.BatchNorm1d(feature_dim),
                           nn.Dropout(p=0.5))
             for i in range(self.num_part)]
        )
        if self.feature_mask:
            self.mask = nn.Sequential(
                    nn.Linear(num_ftrs, feature_dim*n_parts),
                    nn.BatchNorm1d(feature_dim*n_parts),
                    nn.Dropout(p=0.5),
                    nn.Sigmoid())
        #self.branch.apply(weights_init_kaiming)

        if is_training:
            self.classifier = nn.ModuleList(
                [nn.Linear(feature_dim, config["num_labels"])
                 for i in range(self.num_part)]
            )
            if config["asoftmax_params"]["margin"] == 0:
                self.classifier_global = nn.Linear(feature_dim * n_parts,
                        config["num_labels"])
            else:
                import logging
                logging.info("Using angle loss:%s" %
                            config["asoftmax_params"])
                                             
                self.classifier_global = MarginInnerProduct(
                        config, feature_dim * n_parts,
                        config["num_labels"])

    def forward(self, x, labels, return_feature=False):
        x_split = torch.chunk(x, self.num_part, 2)
        x_global = []   # store the concated feature
        y_split = []
        if self.feature_mask:
            mask = F.avg_pool2d(x, kernel_size=x.size()[2:])
            mask = mask.view(mask.size(0), -1)
            mask = self.mask(mask)
        else:
            mask = 1
            
        for i in range(self.num_part):
            x_temp = F.avg_pool2d(x_split[i], kernel_size=x_split[i].size()[2:])
            x_temp_r = x_temp.view(x_temp.size(0), -1)
            x_temp = self.branch[i](x_temp_r)
            if self.training:
                x_global.append(x_temp)
                y_split.append(self.classifier[i](x_temp))
            else:
                y_split.append(x_temp)

        if self.training:
            if self.config["asoftmax_params"]["margin"] > 0:
                y_split.append(self.classifier_global(
                    torch.cat(x_global, dim=1) * mask, labels))
            else:
                y_split.append(self.classifier_global(torch.cat(x_global,
                    dim=1) * mask))
            
            if return_feature:
                return torch.cat(x_global, dim=1) * mask, y_split
            else:
                return y_split
        else:
            return torch.cat(y_split, dim=1) * mask


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
