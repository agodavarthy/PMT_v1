#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:   Travis A. Ebesu
@summary:  BPR Pairwise Generalized Matrix Factorization, Nonlinear variant

r_ui = v^T ReLU(p_u * q_i)

GMF see
He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April).
Neural collaborative filtering.
"""
from util.helper import caffe_init
import torch
import torch.nn as nn
import math
from util.models.mf import MatrixFactorization


class GeneralizedMatrixFactorization(MatrixFactorization):

    def __init__(self, config):
        """
        Nonlinear Variant of GMF

        :param config:
        """
        super(GeneralizedMatrixFactorization, self).__init__(config)
        # Final projection vector
        self.v = nn.Linear(self.config.embed_size, 1, bias=False)
        self.act = nn.ReLU()

        # lecunn uniform
        limit = math.sqrt(3.0 / self.config.embed_size)
        self.v.weight.data.uniform_(-limit, limit)

    def forward(self, user_id, item_id, neg_item_id=None):
        user = self.user_memory(user_id)
        item = self.item_memory(item_id)

        score = self.v(self.act(user * item)).squeeze()
        if neg_item_id is not None:
            neg_item = self.item_memory(neg_item_id)
            neg_score = self.v(self.act(user * neg_item)).squeeze()
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
        return MatrixFactorization.get_l2_reg(self, l2, user_id, item_id, neg_item_id) \
               + l2 * self.v.weight.norm()

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
        # expand_item = self.item_memory(item_id).unsqueeze(1)
        expand_item = self.item_memory.weight.unsqueeze(0)

        elementwise = self.act(expand_item * expand_user)
        shape = elementwise.shape
        return self.v(elementwise.view(-1, shape[-1])).view(shape[0], shape[1])

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

        # [bs, embed] => [bs]
        return self.v(self.act(user_term * item_lf)).squeeze()

    def new_user_recommend(self):
        """
        Perform recommendation over all the items

        :return:
        """
        user_term = self.new_user.unsqueeze(0)
        return self.v(self.act(self.item_memory.weight * user_term)).squeeze()

