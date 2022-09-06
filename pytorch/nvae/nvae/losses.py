from functools import reduce

import numpy as np
import torch
from torch.nn import functional as F

class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        预热KL损失，先对各级别的KL损失进行预热，预热完成后，对M_N的值进行衰减,所有衰减策略采用线性衰减
        :param init_weights: 各级别 KL 损失的初始权重
        :param steps: 各级别KL损失从初始权重增加到1所需的步数
        :param M_N: 初始M_N值
        :param eta_M_N: 最小M_N值
        :param M_N_decay_step: 从初始M_N值到最小M_N值所需的衰减步数
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            # 如果所有级别的KL损失的预热都已完成
            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                # 准备M_N的衰减
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss


def recon(output, target):
    """
    recon loss
    :param output: Tensor. shape = (B, C, H, W)
    :param target: Tensor. shape = (B, C, H, W)
    :return:
    """

    # Treat q(x|z) as Norm distribution
    # loss = F.mse_loss(output, target)

    # Treat q(x|z) as Bernoulli distribution.
    loss = F.binary_cross_entropy(output, target)
    return loss


def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def log_sum_exp(x):
    """

    :param x: Tensor. shape = (batch_size, num_mixtures, height, width)
    :return:
    """

    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m = m2.unsqueeze(1)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=1))


def discretized_mix_logistic_loss(y_hat: torch.Tensor, y: torch.Tensor, num_classes=256, log_scale_min=-7.0):
    """Discretized mix of logistic distributions loss.

    Note that it is assumed that input is scaled to [-1, 1]



    :param y_hat: Tensor. shape=(batch_size, 3 * num_mixtures * img_channels, height, width), predict output.
    :param y: Tensor. shape=(batch_size, img_channels, height, width), Target.
    :return: Tensor loss
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y_hat.chunk(3, dim=1)
    log_scales = torch.clamp_max(log_scales, log_scale_min)

    num_mixtures = y_hat.size(1) // y.size(1) // 3

    B, C, H, W = y.shape
    y = y.unsqueeze(1).repeat(1, num_mixtures, 1, 1, 1).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    log_pdf_mid = min_in - log_scales - 2. * F.softplus(mid_in)

    log_probs = torch.where(y < -0.999, log_cdf_plus,
                            torch.where(y > 0.999, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5, torch.clamp_max(cdf_delta, 1e-12),
                                                    log_pdf_mid - np.log((num_classes - 1) / 2))))

    # (batch_size, num_mixtures * img_channels, height, width)
    log_probs = log_probs + F.softmax(log_probs, dim=1)

    log_probs = [log_sum_exp(log_prob) for log_prob in log_probs.chunk(y.size(1), dim=1)]
    log_probs = reduce(lambda a, b: a + b, log_probs)

    return -torch.sum(log_probs)
