from __future__ import print_function
import torch
import torch.nn as nn


class SupConLossPairNegMask(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLossPairNegMask, self).__init__()
        self.base_temperature = 0.07
        self.temperature = temperature

    def forward(self, features, mask=None):
        """
        :param features: hidden vector of shape [bsz, ...].
        :param mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                comes from the same subject as sample i. Can be asymmetric.
        :return: A loss scalar.
        """
        device = features.device

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)  # torch.Size([16, 128])

        indi_samples = torch.where(mask.sum(1) > 0)[0]
        features = features[indi_samples, :]
        mask = mask[indi_samples, :][:, indi_samples]
        assert torch.eq(mask, mask.T).all()
        batch_size = features.shape[0]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_neg = (mask * log_prob).sum(1) / mask.sum(1)
        lossn = - (self.temperature / self.base_temperature) * mean_log_prob_neg
        lossn = lossn.view(1, batch_size).mean()

        return lossn
