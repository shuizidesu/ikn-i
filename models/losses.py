from typing import List, Dict

from .base_model import KoopmanNet

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self, loss_name: str = "mse"):
        super(BaseLoss, self).__init__()
        self.loss_name = loss_name

    def forward(self, preds: Tensor, labels: Tensor):
        if self.loss_name == "mse":
            loss = F.mse_loss(preds, labels)
        elif self.loss_name == "mae":
            loss = F.l1_loss(preds, labels)
        elif self.loss_name == "nmse":
            loss = (F.mse_loss(preds, labels)) / (torch.square(labels).mean())
        else:
            raise ValueError(f"Loss name {self.loss_name} not implemented!")
        return loss


def k_linear_loss(
        batch_data: Dict[str, Tensor],
        net: KoopmanNet,
        loss_name: str = "mse",
        gamma: float = 0.99
):
    x = batch_data["x"]
    u = batch_data["u"]
    _, steps, _ = x.shape
    x0 = x[:, 0, :]

    base_loss_fn = BaseLoss(loss_name)
    koopman_loss = 0.0
    pred_loss = 0.0
    recon_loss = 0.0
    total_loss = 0.0

    beta = 1.0
    cont = 0.0

    for i in range(steps - 1):
        u0 = u[:, i, :]
        x0_emb = net.x_encoder(x0)
        x0_recon = net.x_decoder(x0_emb)
        u0_emb = net.u_encoder(x0, u0)
        x1_emb_pred = net.koopman_operation(x0_emb, u0_emb)
        x1_pred = net.x_decoder(x1_emb_pred)

        x1 = x[:, i + 1, :]
        x1_emb = net.x_encoder(x1)

        koopman_loss += base_loss_fn(x1_emb_pred, x1_emb)
        pred_loss += base_loss_fn(x1_pred, x1)
        recon_loss += base_loss_fn(x0_recon, x0)
        total_loss += (beta * base_loss_fn(x1_emb_pred, x1_emb) + beta * base_loss_fn(x1_pred, x1)
                       + base_loss_fn(x0_recon, x0) * 0)

        beta *= gamma
        x0 = x1_pred
        cont += 1.0

    return dict(
        total_loss=total_loss / cont,
        koopman_loss=koopman_loss / cont,
        pred_loss=pred_loss / cont,
        recon_loss=recon_loss / cont
    )


def pred_and_eval_loss(
        batch_data: Dict[str, Tensor],
        net: KoopmanNet,
):
    x = batch_data["x"]
    u = batch_data["u"]
    _, steps, _ = x.shape
    x0 = x[:, 0, :]
    pred = x0.unsqueeze(1)

    base_loss_fn = BaseLoss('mae')
    pred_loss = 0.0
    cont = 0.0

    for i in range(steps - 1):
        u0 = u[:, i, :]
        x0_emb = net.x_encoder(x0)
        u0_emb = net.u_encoder(x0, u0)
        x1_emb_pred = net.koopman_operation(x0_emb, u0_emb)
        x1_pred = net.x_decoder(x1_emb_pred)

        x1 = x[:, i + 1, :]

        pred_loss += base_loss_fn(x1_pred, x1)

        x0 = x1_pred
        cont += 1.0

        pred = torch.cat((pred, x1_pred.unsqueeze(1)), dim=1)

    return dict(
        pred=pred,
        pred_loss=pred_loss / cont,
    )
