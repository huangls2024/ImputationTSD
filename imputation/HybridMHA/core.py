"""
The core wrapper assembles the submodules of HybriMHA imputation model
and takes over the forward progress of the algorithm.

"""

# Created by Liusong Huang <012024020443@sgs.msu.edu.my>

from typing import Callable

import torch
import torch.nn as nn

from ...nn.modules.saits import BackboneSAITS
from ...utils.metrics import calc_mae


class _HybridMHA(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        hybrid_attention_mask: bool = True,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        customized_loss_func: Callable = calc_mae,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.hybrid_attention_mask = hybrid_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.customized_loss_func = customized_loss_func

        self.encoder = BackboneSAITS(
            n_steps,
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
        )

    def forward(
        self,
        inputs: dict,
        diagonal_attention_mask: bool = True,
        training: bool = True,
    ) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # determine the attention mask
        if (training and self.diagonal_attention_mask) or (
            (not training) and diagonal_attention_mask
        ):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            # then broadcast on the batch axis
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None


        (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_HybridMHA_attn_weights,
            second_HybridMHA_attn_weights,
            combining_weights,
        ) = self.encoder(X, missing_mask, diagonal_attention_mask)

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * X_tilde_3

        # ensemble the results as a dictionary for return
        results = {
            "first_HybridMHA_attn_weights": first_HybridMHA_attn_weights,
            "second_HybridMHA_attn_weights": second_HybridMHA_attn_weights,
            "combining_weights": combining_weights,
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]

            # calculate loss for the observed reconstruction task (ORT)
            # this calculation is more complicated that pypots.nn.modules.saits.SaitsLoss because
            # SAITS model structure has three parts of representation
            ORT_loss = 0
            ORT_loss += self.customized_loss_func(X_tilde_1, X, missing_mask)
            ORT_loss += self.customized_loss_func(X_tilde_2, X, missing_mask)
            ORT_loss += self.customized_loss_func(X_tilde_3, X, missing_mask)
            ORT_loss /= 3
            ORT_loss = self.ORT_weight * ORT_loss

            # calculate loss for the masked imputation task (MIT)
            MIT_loss = self.MIT_weight * self.customized_loss_func(
                X_tilde_3, X_ori, indicating_mask
            )
            # `loss` is always the item for backward propagating to update the model
            loss = ORT_loss + MIT_loss

            results["ORT_loss"] = ORT_loss
            results["MIT_loss"] = MIT_loss
            results["loss"] = loss

        return results
