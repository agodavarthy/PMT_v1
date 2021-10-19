#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:   Travis A. Ebesu
@summary:  BPR Pairwise MLP
"""
from util.helper import caffe_init
import torch
import torch.nn as nn
import math
from util.models.mf import MatrixFactorization


class MLPMatrixFactorization(MatrixFactorization):

    def __init__(self, config):
        """
        Nonlinear Variant of GMF

        :param config:
        """
        super(MLPMatrixFactorization, self).__init__(config)
        layers = []

        in_size = self.config.embed_size

        for idx in range(self.config.n_layers-1):
            layer = nn.Linear(in_size, in_size // 2)
            # Init
            if getattr(self.config, 'init', 'default') == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)  # Make sure we fire!
            elif getattr(self.config, 'init', 'default') == 'xavier':
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0.01)  # Make sure we fire!
            layers.append(layer)
            layers.append(nn.ReLU())
            in_size = in_size // 2
        layers.append(nn.Linear(in_size, 1, bias=False))
        self.v = nn.Sequential(*layers)
        # lecunn uniform
        limit = math.sqrt(3.0 / self.config.embed_size)
        self.v[-1].weight.data.uniform_(-limit, limit)

    def forward(self, user_id, item_id, neg_item_id=None):
        user = self.user_memory(user_id)
        item = self.item_memory(item_id)

        score = self.v(user * item).squeeze()
        # score = self.v(torch.cat([user, item], dim=1)).squeeze()

        if neg_item_id is not None:
            raise NotImplementedError
            neg_item = self.item_memory(neg_item_id)
            neg_score = self.v(torch.cat([user, neg_item], dim=1)).squeeze()
            return score, neg_score
        return score

    def get_l2_reg(self, l2: float, user_id: torch.LongTensor,
                   item_id: torch.LongTensor,
                   neg_item_id: torch.LongTensor):
        """
        Get L2 regularization for Pairwise,
        Passing it in optimizer for weight decay does not seem to perform as well
        for some reason. Possibly the positive and negative sampling?

        :param l2:
        :param user_id:
        :param item_id:
        :param neg_item_id:
        :return:
        """
        reg = 0.0
        for p in self.v.parameters():
            reg += l2 * p.norm()
        return MatrixFactorization.get_l2_reg(self, l2, user_id, item_id, neg_item_id) \
               + reg

    def recommend(self, user_id):
        """
        Given a Tensor of item_ids return the recommendation scores for all
        users for the given item_id

        :param user_id: [batch size]
        :return: [batch size, n_users]
        """
        # [1, n users, embed]
        # expand_user = self.user_memory.weight.unsqueeze(0)
        expand_user = self.user_memory(user_id).unsqueeze(1)
        # [bs, 1, embed] ==> [bs, n users, embed]
        expand_item = self.item_memory.weight.unsqueeze(0)

        # interaction = torch.cat([expand_user, expand_item], dim=-1)
        interaction = expand_user * expand_item
        shape = interaction.shape
        return self.v(interaction.view(-1, shape[-1])).view(shape[0], shape[1])

    def new_user_score(self, item_lf: torch.FloatTensor):
        """
        Given the new weighted item latent factor compute this with the new
        user latent factor.

        :param item_lf: item latent factor, [batch, embed size] or [embed size]
        :return: Recommendation score
        """
        # [1, embed]
        user_term = self.new_user.unsqueeze(0)

        if item_lf.dim() == 1:
            item_lf = item_lf.unsqueeze(0)
        # interaction = torch.cat([user_term, item_lf], dim=1)
        interaction = user_term * item_lf
        # [bs, embed] => [bs]
        return self.v(interaction).squeeze()

    def new_user_recommend(self):
        """
        Perform recommendation over all the items

        :return:
        """
        return self.new_user_score(self.item_memory.weight)

