from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as torch_nn
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset

from ..evaluation import calc_expected_percentile_rank, calc_tpr
from ..utils import CategoryDict, paths

EPS = 1e-6
PATH_DIR_DATA_PROCESS = Path("data_processed")


class FossilNOW(Dataset):
    def __init__(self, data: DataFrame, device: str = "cpu") -> None:
        super().__init__()

        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index]

        occurence = torch.tensor(
            x["occurence"], device=self.device, dtype=torch.float32
        )
        sites = torch.tensor(x["site"], device=self.device, dtype=torch.int32)
        genera = torch.tensor(x["genus"], device=self.device, dtype=torch.int32)

        return occurence, sites, genera


class MF(torch_nn.Module):
    def __init__(
        self,
        n_sites: int,
        n_genera: int,
        d_hid: int = 64,
        prob_output: bool = False,
        is_layernorm: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.device = device
        self.prob_output = prob_output
        self.is_layernorm = is_layernorm
        self.n_sites, self.n_genera = n_sites, n_genera

        self.emd_site = torch_nn.Embedding(n_sites, d_hid)
        self.embd_genera = torch_nn.Embedding(n_genera, d_hid)
        self.batchnorm = torch_nn.BatchNorm2d(1)
        self.layernorm = torch_nn.LayerNorm(d_hid)

        self.act_prob = torch_nn.Sigmoid()

    def forward(self, idx_sites: Tensor, idx_genera: Tensor):
        emb_sites = self.emd_site(idx_sites)
        # [bz, n_sites_pack, d_hid]
        emb_genera = self.embd_genera(idx_genera)
        # [bz, n_genera_pack, d_hid]

        if self.is_layernorm is True:
            emb_sites = self.layernorm(emb_sites)
            emb_genera = self.layernorm(emb_genera)

        emb_genera_T = torch.permute(emb_genera, (0, 2, 1))
        # [bz, d_hid, n_genera_pack]

        occurence_pred = torch.bmm(emb_sites, emb_genera_T)
        # [bz, n_sites_pack, n_genera_pack]

        if self.prob_output is True:
            occurence_pred = self.act_prob(occurence_pred)

        return occurence_pred

    def get_embds(self) -> tuple[ndarray, ndarray]:
        idx_sites = torch.arange(
            0,
            self.n_sites,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)
        # [1, n_sites]
        idx_genera = torch.arange(
            0,
            self.n_genera,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(0)
        # [1, n_genera]

        emb_sites = self.emd_site(idx_sites)
        # [bz, n_sites_pack, d_hid]
        emb_genera = self.embd_genera(idx_genera)
        # [bz, n_genera_pack, d_hid]

        if self.is_layernorm is True:
            emb_sites = self.layernorm(emb_sites)
            emb_genera = self.layernorm(emb_genera)

        emb_sites = emb_sites.detach().cpu().numpy()
        emb_genera = emb_genera.detach().cpu().numpy()

        return emb_sites, emb_genera


def train(
    model: Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion,
    scheduler,
):
    model.train()

    for x in loader:
        optimizer.zero_grad()

        occurence, idx_sites, idx_genera = x

        pred = model(idx_sites, idx_genera)
        loss = calc_loss(criterion, occurence, pred)

        loss.backward()
        optimizer.step()

    scheduler.step()


def validate(
    model: Module,
    loader: DataLoader,
    criterion,
):
    preds = []

    model.eval()
    with torch.no_grad():
        losses = []
        for x in loader:
            occurence, idx_sites, idx_genera = x

            pred = model(idx_sites, idx_genera)
            loss = calc_loss(criterion, occurence, pred)

            preds.append(
                {
                    "site": idx_sites.detach().cpu().numpy(),
                    "genus": idx_genera.detach().cpu().numpy(),
                    "occurence": occurence.detach().cpu().numpy(),
                    "prediction": pred.detach().cpu().numpy(),
                }
            )

            losses.append(loss.item())

    loss = sum(losses) / len(losses)

    return loss, preds


def calc_loss(criterion, occ: torch.Tensor, pred: torch.Tensor, alpha: float = 10):
    loss = criterion(occ, pred)

    confidence = 1 + alpha * occ
    loss = torch.mean(confidence * loss)

    return loss


def trigger_train(
    df_train: DataFrame,
    df_test: DataFrame,
    device: str = "cpu",
    use_reg: bool = True,
    num_epochs: int = 10,
    lr: float = 5e-3,
    batch_size: int = 25,
    dim_hid: int = 10,
    output_prob: bool = False,
    num_sites_per_pack: int = 2,
    num_genera_per_pack: int = 4,
) -> DataFrame:
    """Start training

    Args:
        df_train (DataFrame): training
        df_test (DataFrame): testing dataframe
        device (str, optional): working device. Defaults to "cpu".
        use_reg (bool, optional): _description_. Defaults to True.
        num_epochs (int, optional): _description_. Defaults to 10.
        lr (float, optional): _description_. Defaults to 5e-3.
        batch_size (int, optional): _description_. Defaults to 25.
        dim_hid (int, optional): _description_. Defaults to 10.
        output_prob (bool, optional): _description_. Defaults to False.
        num_sites_per_pack (int, optional): _description_. Defaults to 2.
        num_genera_per_pack (int, optional): _description_. Defaults to 4.

    Returns:
        DataFrame: predicted dataframe, predict both train and test data
    """

    # Load necessary things
    enc_genera = CategoryDict.from_file(paths.encoding_genera)
    enc_site = CategoryDict.from_file(paths.encoding_sites)
    dataset_train, dataset_val = FossilNOW(df_train), FossilNOW(df_test)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    mf = MF(
        enc_site.size(),
        enc_genera.size(),
        d_hid=dim_hid,
        prob_output=output_prob,
    )
    # mf.load_state_dict(torch.load("model_best_PROBABILITY_OUTPUT=False_0.1747.pt"))
    mf = mf.to(device=torch.device(device))

    optimizer = AdamW(mf.parameters(), lr=lr, weight_decay=2e-5 if use_reg else 0)
    criterion = torch_nn.MSELoss(reduction="none")
    scheduler = lr_scheduler.LinearLR(optimizer, 1.0, 5e-2, batch_size)

    best_loss_val = 10e10
    best_state_dict = None
    best_emb_sites, best_emb_genera = None, None

    for n in range(num_epochs):
        print(f"== Epoch: {n:02d}")

        # Train
        train(mf, loader_train, optimizer, criterion, scheduler)

        # Validate
        loss_val, _ = validate(mf, loader_val, criterion)

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            best_state_dict = mf.state_dict()
            best_emb_sites, best_emb_genera = mf.get_embds()
        else:
            break

        print(f"Loss val: {loss_val}")

    # Save embedding and model
    if output_prob is True:
        path_emb_sites = paths.embd_sites_prob_true
        path_emb_genera = paths.embd_genera_prob_true
        path_model = paths.weight_mf_prob_true
    else:
        path_emb_sites = paths.embd_sites_prob_false
        path_emb_genera = paths.embd_genera_prob_false
        path_model = paths.weight_mf_prob_false
    assert best_emb_sites is not None
    assert best_emb_genera is not None

    np.save(path_emb_sites, best_emb_sites)
    np.save(path_emb_genera, best_emb_genera)

    torch.save(best_state_dict, path_model)

    # Predict entire dataset
    _, preds_train = validate(mf, loader_train, criterion)
    _, preds_val = validate(mf, loader_val, criterion)
    preds = [*preds_train, *preds_val]

    list_preds = []
    for pred in preds:
        for sites, genera, _, pre in zip(
            pred["site"], pred["genus"], pred["occurence"], pred["prediction"]
        ):
            for i in range(num_sites_per_pack):
                for j in range(num_genera_per_pack):
                    list_preds.append(
                        {
                            "site": enc_site.ids2names(sites[i].item())[0],
                            "genus": enc_genera.ids2names(genera[j].item())[0],
                            "pred": pre[i, j],
                        }
                    )

    df_out = pd.DataFrame.from_records(list_preds).pivot(
        index="site", columns="genus", values="pred"
    )

    return df_out


def trigger_test(
    df_test: DataFrame,
    device: str = "cpu",
    batch_size: int = 25,
    dim_hid: int = 10,
    output_prob: bool = False,
    num_sites_per_pack: int = 2,
    num_genera_per_pack: int = 4,
):
    # Load model and other things
    enc_genera = CategoryDict.from_file(paths.encoding_genera)
    enc_site = CategoryDict.from_file(paths.encoding_sites)

    if output_prob is True:
        path_model = paths.weight_mf_prob_true
    else:
        path_model = paths.weight_mf_prob_false
    mf = MF(
        enc_site.size(),
        enc_genera.size(),
        d_hid=dim_hid,
        prob_output=output_prob,
    )
    mf.load_state_dict(torch.load(path_model))
    mf.to(device)

    criterion = torch_nn.MSELoss(reduction="none")

    loader_val = DataLoader(FossilNOW(df_test), batch_size=batch_size, shuffle=False)

    # Start validate
    _, preds = validate(mf, loader_val, criterion)

    list_preds = []
    for pred in preds:
        for sites, genera, occurence, pre in zip(
            pred["site"], pred["genus"], pred["occurence"], pred["prediction"]
        ):
            for i in range(num_sites_per_pack):
                for j in range(num_genera_per_pack):
                    list_preds.append(
                        {
                            "site": enc_site.ids2names(sites[i].item())[0],
                            "genus": enc_genera.ids2names(genera[j].item())[0],
                            "pred": pre[i, j],
                            "occurence": occurence[i, j],
                        }
                    )

    # Calculate metrics
    df_pred = pd.DataFrame.from_records(list_preds)

    if output_prob is True:
        tpr = calc_tpr(df_pred)
    else:
        tpr = -1

    return {
        "expected_percentile_rank": calc_expected_percentile_rank(df_pred),
        "true_positive_rate": tpr,
    }
