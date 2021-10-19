import torch
import torch.nn as nn
from util.helper import caffe_init
from util.models.base import BasePretrainedModel


class MatrixFactorization(BasePretrainedModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.user_memory = nn.Embedding(self.config.user_count,
                                        self.config.embed_size)
        self.item_memory = nn.Embedding(self.config.item_count,
                                        self.config.embed_size)
        caffe_init(self.user_memory.weight)
        caffe_init(self.item_memory.weight)
        self._user_mean = self.user_memory.weight.data.mean(0).squeeze()
        var = self.user_memory.weight.data.var(0).squeeze()
        self._user_dist = torch.distributions.Uniform(-var, var)

        self.new_user = nn.Parameter(torch.Tensor(self.config.embed_size))
        caffe_init(self.new_user)

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
        return l2 * self.item_memory(item_id).norm() \
               + l2 * self.item_memory(neg_item_id).norm() \
               + l2 * self.user_memory(user_id).norm()

    def new_user_reset(self, method='random'):
        """Reset the new user parameters"""
        with torch.no_grad():
            if method == 'random':
                caffe_init(self.new_user)
            elif method == 'mean':
                self.new_user.data = self._user_mean.data
            elif method == 'var':
                # Uniform according to the variance
                self.new_user.data = self._user_dist.sample()
                # self.new_user.data.uniform_(-self._user_var.data, self._user_var.data)
            else:
                raise ValueError(f"Unknown initialization scheme: {method}")

    def new_user_recommend(self):
        """
        Perform recommendation over all the items
        :return:
        """
        user_term = self.new_user.unsqueeze(0)
        return (self.item_memory.weight * user_term).sum(1)

    def new_user_score(self, item_lf: torch.FloatTensor):
        """
        Given the new weighted item latent factor compute this with the new
        user latent factor.

        :param item_lf: item latent factor, [batch, embed size] or [embed size]
        :return: Recommendation score
        """
        # [1, embed]
        user_term = self.new_user.unsqueeze(0)
        # [bs, embed] => [bs]
        return (user_term * item_lf).sum(1)

    def recommend(self, user_id: torch.LongTensor):
        """
        Given a Tensor of user_ids return the recommendation scores for all
        the items given the users

        :param user_id: [batch size]
        :return: [batch size, n_users]
        """
        # [1, n users, embed]
        expand_user = self.user_memory(user_id).unsqueeze(1)

        # [bs, 1, embed] ==> [bs, n users, embed]
        expand_item = self.item_memory.weight.unsqueeze(0)

        # [bs, n_users]
        return (expand_item * expand_user).sum(2)

    def forward(self, user_id, item_id, neg_item_id=None):
        user = self.user_memory(user_id)
        item = self.item_memory(item_id)
        # [bs, embed] [embed, bs] => [bs]
        # score = torch.matmul(item, user.transpose(1, 0)).squeeze()
        # Doing elementwise mult and sum is faster than transpose + matmult
        score = (user * item).sum(1).squeeze()

        if neg_item_id is not None:
            neg_item = self.item_memory(neg_item_id)
            # neg_score = torch.matmul(neg_item, user.transpose(1, 0)).squeeze()
            # Doing elementwise mult and sum is faster than transpose + matmult
            neg_score = (user * neg_item).sum(1).squeeze()
            return score, neg_score
        return score

    def new_user_parameters(self):
        """
        List of torch tensor parameters for the new user

        :return:
        """
        return [self.new_user]


class MatrixFactorization_OLD(nn.Module):
    def __init__(self, config):
        super(MatrixFactorization_OLD, self).__init__()
        self.config = config
        self.user_memory = nn.Embedding(self.config.user_count,
                                        self.config.embed_size)
        self.item_memory = nn.Embedding(self.config.item_count,
                                        self.config.embed_size)

        caffe_init(self.user_memory.weight)
        caffe_init(self.item_memory.weight)

        self.new_user = nn.Parameter(torch.Tensor(self.config.embed_size))
        nn.init.normal_(self.new_user.data, std=0.01)
        self._user_mean = None
        self._user_std = None

    def new_user_reset(self, mean: torch.FloatTensor, std: torch.FloatTensor):
        """
        Reset the user latent factor

        :param mean:
        :param std:
        :return:
        """
        if not mean.shape == std.shape == self.new_user.shape:
            print(f"Invalid mean/std shape should be {self.new_user.shape}")
            raise Exception(f"Invalid mean/std shape should be {self.new_user.shape}")
        self._user_mean = mean
        self._user_std = std
        # Init to normal with same mean/std
        self.new_user.data = torch.normal(mean, std).to(next(self.parameters()).device)

    def forward(self, user_id, item_id, neg_item_id=None):
        user = self.user_memory(user_id)
        item = self.item_memory(item_id)
        # [bs, embed] [embed, bs] => [bs]
        # score = torch.matmul(item, user.transpose(1, 0)).squeeze()
        # Doing elementwise mult and sum is faster than transpose + matmult
        score = (user * item).sum(1).squeeze()

        if neg_item_id is not None:
            neg_item = self.item_memory(neg_item_id)
            # neg_score = torch.matmul(neg_item, user.transpose(1, 0)).squeeze()
            # Doing elementwise mult and sum is faster than transpose + matmult
            neg_score = (user * neg_item).sum(1).squeeze()
            return score, neg_score
        return score

    def recommend(self, item_id):
        """
        Given a Tensor of item_ids return the recommendation scores for all
        users for the given item_id

        :param item_id: [batch size]
        :return: [batch size, n_users]
        """
        # [1, n users, embed]
        expand_user = self.user_memory(item_id).unsqueeze(1)

        # [bs, 1, embed] ==> [bs, n users, embed]
        expand_item = self.item_memory.weight.unsqueeze(0)

        # [bs, n_users]
        return (expand_item * expand_user).sum(2)

    def get_weighted_item_lf(self, sim_scores: torch.FloatTensor, item_id: torch.LongTensor):
        """

        :param sim_scores:
        :param item_id:
        :return:
        """
        # [bs, embed]
        item_lf = self.item_memory(item_id)

        # attention, softmax normalize
        attn = nn.functional.softmax(sim_scores, -1)

        # Broadcast and weighted sum
        item_latent_factor = (attn.unsqueeze(-1) * item_lf).sum(0)
        return item_latent_factor

    def new_user_recommend(self, method: str):
        """
        Perform recommendation over all the items

        :return:
        """
        user_term = self.new_user.unsqueeze(0)
        if method == 'inner':
            user_term = user_term
        elif method == 'offset':
            user_term = (self._user_mean.data.view(1, -1) + user_term)
        else:
            raise Exception("Unknown method: %s" % method)
        return (self.item_memory.weight * user_term).sum(1)

    def new_user_score(self, item_lf: torch.FloatTensor, method: str):
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
        if method == 'inner':
            user_term = user_term
        elif method == 'offset':
            user_term = (self._user_mean.data.view(1, -1) + user_term)
        else:
            raise Exception("Unknown method: %s" % method)

        # [bs, embed] => [bs]
        return (user_term * item_lf).sum(1)


class ItemBiasMatrixFactorization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.user_memory = nn.Embedding(self.config.user_count,
                                        self.config.embed_size)
        self.item_memory = nn.Embedding(self.config.item_count,
                                        self.config.embed_size)
        self.item_bias = nn.Embedding(self.config.item_count, 1)
        # Final projection vector
        nn.init.normal_(self.user_memory.weight, std=0.01)
        nn.init.normal_(self.item_memory.weight, std=0.01)
        nn.init.constant_(self.item_bias.weight, 0.0)
        self.new_user = nn.Parameter(torch.Tensor(self.config.embed_size))
        nn.init.normal_(self.new_user.data, std=0.01)
        self._user_mean = None
        self._user_std = None

    def forward(self, user_id, item_id, neg_item_id=None):
        user = self.user_memory(user_id)
        item_bias = self.item_bias(item_id)
        item = self.item_memory(item_id)
        # [bs, embed] [embed, bs] => [bs]
        # score = torch.matmul(item, user.transpose(1, 0)).squeeze()
        # Doing elementwise mult and sum is faster than transpose + matmult
        score = (user * item).sum(1).squeeze() + item_bias.squeeze()

        if neg_item_id is not None:
            neg_item_bias = self.item_bias(neg_item_id)
            neg_item = self.item_memory(neg_item_id)
            # neg_score = torch.matmul(neg_item, user.transpose(1, 0)).squeeze()
            # Doing elementwise mult and sum is faster than transpose + matmult
            neg_score = (user * neg_item).sum(1).squeeze() + neg_item_bias.squeeze()
            return score, neg_score
        return score

    def recommend(self, user_id):
        """
        Given a Tensor of item_ids return the recommendation scores for all
        users for the given item_id

        :param user_id: [batch size]
        :return: [batch size, n_users]
        """
        # [n users, 1, embed]
        expand_user = self.user_memory(user_id).unsqueeze(1)

        # [1, n_items, embed]
        expand_item = self.item_memory.weight.unsqueeze(0)

        # ==> [n_users, n items, embed] ==> [n_users, n_items] + [n_items, 1]
        return (expand_item * expand_user).sum(2) + self.item_bias.weight.view(1, -1)

    def new_user_reset(self, mean: torch.FloatTensor, std: torch.FloatTensor):
        """
        Reset the user latent factor

        :param mean:
        :param std:
        :return:
        """
        if not mean.shape == std.shape == self.new_user.shape:
            print(f"Invalid mean/std shape should be {self.new_user.shape}")
            raise Exception(f"Invalid mean/std shape should be {self.new_user.shape}")
        self._user_mean = mean
        self._user_std = std
        # Init to normal with same mean/std
        self.new_user.data = torch.normal(mean, std).to(next(self.parameters()).device)

    # def get_weighted_item_lf(self, sim_scores: torch.FloatTensor, item_id: torch.LongTensor):
    #     """
    #
    #     :param sim_scores:
    #     :param item_id:
    #     :return:
    #     """
    #     # [bs, embed]
    #     item_lf = self.item_memory(item_id)
    #
    #     # attention, softmax normalize
    #     attn = nn.functional.softmax(sim_scores, -1)
    #
    #     # Broadcast and weighted sum
    #     item_latent_factor = (attn.unsqueeze(-1) * item_lf).sum(0)
    #     return item_latent_factor

    def new_user_recommend(self, method: str):
        """
        Perform recommendation over all the items

        :return:
        """
        user_term = self.new_user.unsqueeze(0)
        if method == 'inner':
            user_term = user_term
        elif method == 'offset':
            user_term = (self._user_mean.data.view(1, -1) + user_term)
        else:
            raise Exception("Unknown method: %s" % method)
        return (self.item_memory.weight * user_term).sum(1) + self.item_bias.weight.squeeze()

    def new_user_score(self, item_lf: torch.FloatTensor, method: str):
        """
        Given the new weighted item latent factor compute this with the new
        user latent factor.

        We cant use an item bias unless we also weight it

        :param item_lf: item latent factor, [batch, embed size] or [embed size]
        :return: Recommendation score
        """
        # [1, embed]
        user_term = self.new_user.unsqueeze(0)

        if item_lf.dim() == 1:
            item_lf = item_lf.unsqueeze(0)
        if method == 'inner':
            user_term = user_term
        elif method == 'offset':
            user_term = (self._user_mean.data.view(1, -1) + user_term)
        else:
            raise Exception("Unknown method: %s" % method)

        # [bs, embed] => [bs]
        return (user_term * item_lf).sum(1)

# class BPRMF(nn.Module):
#
#     def __init__(self, config):
#         """
#         config.user_count, config.item_count, config.embed_size
#
#         :param config:
#         """
#         super(BPRMF, self).__init__()
#         self.config = config
#         self.user_memory = nn.Embedding(self.config.user_count,
#                                         self.config.embed_size)
#         self.item_memory = nn.Embedding(self.config.item_count,
#                                         self.config.embed_size)
#         nn.init.normal_(self.user_memory.weight, std=0.01)
#         nn.init.normal_(self.item_memory.weight, std=0.01)
#         self.new_user = nn.Parameter(torch.Tensor(self.config.embed_size))
#         nn.init.normal_(self.new_user.data, std=0.01)
#
#     def forward(self, user_id: torch.LongTensor, item_id: torch.LongTensor,
#                 neg_item_id: torch.LongTensor=None):
#         # [B, embed_size]
#         user = self.user_memory(user_id)
#         item = self.item_memory(item_id)
#
#         score = torch.matmul(user, item.transpose(1, 0)).squeeze()
#         if neg_item_id is not None:
#             neg_item = self.item_memory(neg_item_id)
#             neg_score = torch.matmul(user, neg_item.transpose(1, 0)).squeeze()
#             return score, neg_score
#         return score
#
#     def recommend(self, item_id: torch.LongTensor):
#         # [n users, embed]
#         user = self.user_memory.weight
#
#         # [bs, embed]
#         item = self.item_memory(item_id)
#         # [bs, embed] [embed, n_users] => [bs, n users]
#         return torch.matmul(item, user.transpose(1, 0))
#
#     def new_user_reset(self, mean: torch.FloatTensor, std=torch.FloatTensor):
#         """
#         Reset the user latent factor
#
#         :param mean:
#         :param std:
#         :return:
#         """
#         if not mean.shape == std.shape == self.new_user.shape:
#             print(f"Invalid mean/std shape should be {self.new_user.shape}")
#             raise Exception(f"Invalid mean/std shape should be {self.new_user.shape}")
#         # Init to normal with same mean/std
#         self.new_user.data = torch.normal(mean, std)
#
#     def new_user_score(self, item_id: torch.LongTensor):
#         """
#         Compute ranking score
#
#         :param item_id:
#         :return:
#         """
#         # [bs, embed]
#         item = self.item_memory(item_id)
#         # new_user [bs, embed] [embed, 1] => [bs]
#         return torch.matmul(item, self.new_user.unsqueeze(1)).squeeze()
