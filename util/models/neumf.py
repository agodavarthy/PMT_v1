#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:   Travis A. Ebesu
@summary:  see

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April).
Neural collaborative filtering.
"""
from util.helper import caffe_init
import torch
import torch.nn as nn
import math
from util.models.base import BasePretrainedModel


class NeuralMatrixFactorization(BasePretrainedModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mf_user = nn.Embedding(self.config.user_count,
                                    self.config.embed_size)
        self.mf_item = nn.Embedding(self.config.item_count,
                                    self.config.embed_size)

        caffe_init(self.mf_user.weight)
        caffe_init(self.mf_item.weight)
        # Divide by two so we can concat them for one pass in mlp
        in_size = self.config.layers[0] // 2
        if self.config.tie:
            self.mlp_user = self.mf_user
            self.mlp_item = self.mf_item
        else:
            self.mlp_user = nn.Embedding(self.config.user_count,
                                         in_size)
            self.mlp_item = nn.Embedding(self.config.item_count,
                                         in_size)
        mlp = []

        for idx in range(1, len(self.config.layers)):
            mlp.append(nn.Linear(self.config.layers[idx-1], self.config.layers[idx]))
            mlp.append(nn.ReLU())

        self.mlp = nn.Sequential(*mlp)

        self.v = nn.Linear(self.config.embed_size + self.config.layers[-1], 1, bias=False)
        # lecunn uniform
        limit = math.sqrt(3.0 / self.config.embed_size)
        self.v.weight.data.uniform_(-limit, limit)

        self.mlp_new_user = nn.Parameter(torch.Tensor(self.config.layers[0] // 2))
        self.mf_new_user = nn.Parameter(torch.Tensor(self.config.embed_size))
        caffe_init(self.mlp_new_user)
        caffe_init(self.mf_new_user)

        self._user_mf_mean = self.mf_user.weight.data.mean(0).squeeze()
        var = self.mf_user.weight.data.var(0).squeeze()
        self._user_mf_dist = torch.distributions.Uniform(-var, var)

        self._user_mlp_mean = self.mlp_user.weight.data.mean(0).squeeze()
        var = self.mlp_user.weight.data.var(0).squeeze()
        self._user_mlp_dist = torch.distributions.Uniform(-var, var)

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
        for p in self.mlp.parameters():
            reg += l2 * p.norm()
        return reg \
               + l2 * self.mf_item(item_id).norm() \
               + l2 * self.mlp_item(item_id).norm() \
               + l2 * self.mf_item(neg_item_id).norm() \
               + l2 * self.mlp_item(neg_item_id).norm() \
               + l2 * self.mf_user(user_id).norm() \
               + l2 * self.mlp_user(user_id).norm() \
               + l2 * self.v.weight.norm()

    def new_user_reset(self, method='random'):
        """Reset the new user parameters"""
        with torch.no_grad():
            if method == 'random':
                caffe_init(self.mlp_new_user)
                caffe_init(self.mf_new_user)
            elif method == 'mean':
                self.mlp_new_user.data = self._user_mlp_mean.data
                self.mf_new_user.data = self._user_mf_mean.data
            elif method == 'var':
                self.mlp_new_user.data = self._user_mlp_dist.sample()
                self.mf_new_user.data = self._user_mf_dist.sample()
                # self.new_user.data.uniform_(-self._user_var.data, self._user_var.data)
            else:
                raise ValueError(f"Unknown initialization scheme: {method}")

    def forward(self, user_id, item_id, neg_item_id=None):
        mf_user = self.mf_user(user_id)
        mf_item = self.mf_item(item_id)
        mf_vector = mf_user * mf_item
        # Concat user and item MLP embed
        mlp_input = torch.cat([self.mlp_user(user_id), self.mlp_item(item_id)], dim=1)
        # Concat MF vector and MLP outputs
        join = torch.cat([mf_vector, self.mlp(mlp_input)], dim=1)
        # Project to a score
        score = self.v(join).squeeze()
        return score

    def new_user_parameters(self):
        """
        List of torch tensor parameters for the new user

        :return:
        """
        return [self.mlp_new_user, self.mf_new_user]

    def recommend(self, user_id: torch.LongTensor):
        """
        Given a Tensor of user_ids return the recommendation scores for all
        the items given the users

        :param user_id: [batch size]
        :return: [batch size, n_users]
        """
        # [batch users, 1, embed]
        mf_user = self.mf_user(user_id).unsqueeze(1)

        # [n_items, embed] => [1, n items, embed]
        mf_item = self.mf_item.weight.unsqueeze(0)

        # [batch users, n items, embed]
        mf_vector = mf_user * mf_item

        # [n items, embed/2]
        mlp_item = self.mlp_item.weight
        # [n users, embed/2]
        mlp_user = self.mlp_user(user_id)
        n_items = self.mlp_item.num_embeddings
        scores = []
        for i, user in enumerate(mlp_user):
            # [1, embed size/2] = expand => [n_items, embed size/2] = cat => [n_items, embed size]
            mlp_input = torch.cat([user.unsqueeze(0).expand(n_items, -1), mlp_item], dim=1)
            # [n items]
            scores.append(self.v(torch.cat([mf_vector[i], self.mlp(mlp_input)], dim=1)).squeeze())
        # [batch users, n items]
        return torch.stack(scores)

    def new_user_score(self, mf_item_lf: torch.FloatTensor, mlp_item_lf: torch.FloatTensor):
        """
        We use the same item latent factors for both the MF and MLP parts which is not correct

        :param mf_item_lf: item latent factor, [batch, embed size] or [embed size]
        :return: Recommendation score
        """
        # [1, embed]
        user_mlp = self.mlp_new_user.unsqueeze(0)
        user_mf = self.mf_new_user.unsqueeze(0)

        # [bs, embed size]
        if mf_item_lf.dim() == 1:
            mf_item_lf = mf_item_lf.unsqueeze(0)

        if mlp_item_lf.dim() == 1:
            mlp_item_lf = mlp_item_lf.unsqueeze(0)

        mf_vector = user_mf * mf_item_lf
        # Concat user and item MLP embed
        # user is repeated for each item
        mlp_input = torch.cat([user_mlp.expand_as(mlp_item_lf), mlp_item_lf], dim=1)
        # Concat MF vector and MLP outputs
        join = torch.cat([mf_vector, self.mlp(mlp_input)], dim=1)

        # [bs, embed] => [bs]
        return self.v(join).squeeze()

    def new_user_recommend(self):
        """
        Perform recommendation over all the items

        :return:
        """
        return self.new_user_score(self.mf_item.weight, self.mlp_item.weight)


if __name__ == '__main__':

    from argparse import Namespace
    model = NeuralMatrixFactorization(Namespace(user_count=5, item_count=30, embed_size=8, layers=[16,8,4]))
    output = model.recommend(torch.LongTensor([0, 1]))
    output.shape

    import torch.nn as nn
    import torch
    from hessian_eigenthings import compute_hessian_eigenthings
    import torch.utils.data
    import numpy as np

    # model(torch.tensor([[2.], [3.]]))
    x = torch.from_numpy(np.random.normal(size=(8, 2))).float()
    y = torch.from_numpy(np.random.normal(size=(8, 1))).float()
    dataloader = torch.utils.data.TensorDataset(x, y)
    top_eigen = 20

    eigenvals, eigenvecs = compute_hessian_eigenthings(
            nn.Linear(2, 1), dataloader,
            loss=nn.functional.mse_loss,
            num_eigenthings=top_eigen, use_gpu=False)
    eigenvals