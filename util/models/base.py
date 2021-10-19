import torch
from torch import nn as nn
from typing import List


class BasePretrainedModel(nn.Module):
    """
    Functions signatures to implement to use with simulator
    also must implement
    def forward(user_id, item_id) -> Recommendation Scores

    This is the functions we use to mainly to reset the new_user and perform its
    recommendations.
    """

    def new_user_reset(self):
        """Reset the new user parameters"""
        raise NotImplementedError

    def new_user_recommend(self):
        """
        Perform recommendation over all the items
        :return:
        """
        raise NotImplementedError

    def new_user_score(self, item_lf: torch.FloatTensor):
        """
        Given the new weighted item latent factor compute this with the new
        user latent factor.

        :param item_lf: item latent factor, [batch, embed size] or [embed size]
        :return: Recommendation score
        """
        raise NotImplementedError

    def recommend(self, user_id: torch.LongTensor):
        """
        Given a Tensor of user_ids return the recommendation scores for all
        the items given the users

        :param user_id: [batch size]
        :return: [batch size, n_users]
        """
        raise NotImplementedError

    def new_user_parameters(self) -> List[torch.Tensor]:
        """
        List of torch tensor parameters for the new user

        :return:
        """
        raise NotImplementedError

    def freeze_params(self):
        """Freezes all parameters except for the new user parameters to learn"""
        print()
        new_user = set(self.new_user_parameters())
        for name, p in self.named_parameters():
            if p in new_user:
                print(f"[Keeping Parameter Trainable: {name}]")
                continue
            if 'user' in name or 'item' in name:
                l2 = p.norm(dim=1)
                l2_std = l2.std()
                l2 = l2.mean()
                min_val = p.min(dim=1)[0].mean()
                max_val = p.max(dim=1)[0].mean()
            else:
                l2 = p.norm()
                l2_std = torch.tensor([0.0])
                min_val = p.min()
                max_val = p.max()
            mean = p.mean()
            std = p.std()

            # Weight statistics
            s = f"{name:<20} L2: {l2.item():>8.4f} ± {l2_std.item():<10.2f} " \
                f"Mean: {mean.item():>8.4f} ± {std.item():<10.2f} " \
                f"Min: {min_val.item():<10.4f} " \
                f"Max: {max_val.item():<10.4f}"
            print(f"[Freezing Parameter: {s}]")
            p.requires_grad = False
        print()

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
        raise NotImplementedError

