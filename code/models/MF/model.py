import torch
import torch.nn as torch_nn
from numpy import ndarray
from torch import Tensor

DEVICE = torch.device("mps")
EPS = 1e-6


class MF(torch_nn.Module):
    def __init__(
        self,
        n_sites: int,
        n_species: int,
        d_hid: int = 64,
        prob_output: bool = False,
        is_layernorm: bool = False,
    ) -> None:
        super().__init__()

        self.prob_output = prob_output
        self.is_layernorm = is_layernorm
        self.n_sites, self.n_species = n_sites, n_species

        self.emd_site = torch_nn.Embedding(n_sites, d_hid)
        self.emd_species = torch_nn.Embedding(n_species, d_hid)
        self.batchnorm = torch_nn.BatchNorm2d(1)
        self.layernorm = torch_nn.LayerNorm(d_hid)

        self.act_prob = torch_nn.Sigmoid()

    def forward(self, idx_sites: Tensor, idx_species: Tensor):
        emb_sites = self.emd_site(idx_sites)
        # [bz, n_sites_pack, d_hid]
        emb_species = self.emd_species(idx_species)
        # [bz, n_species_pack, d_hid]

        if self.is_layernorm is True:
            emb_sites = self.layernorm(emb_sites)
            emb_species = self.layernorm(emb_species)

        emb_species_T = torch.permute(emb_species, (0, 2, 1))
        # [bz, d_hid, n_species_pack]

        occurence_pred = torch.bmm(emb_sites, emb_species_T)
        # [bz, n_sites_pack, n_species_pack]

        if self.prob_output is True:
            occurence_pred = self.act_prob(occurence_pred)

        return occurence_pred

    def get_embds(self) -> tuple[ndarray, ndarray]:
        idx_sites = torch.arange(
            0,
            self.n_sites,
            dtype=torch.int32,
            device=DEVICE,
        ).unsqueeze(0)
        # [1, n_sites]
        idx_species = torch.arange(
            0,
            self.n_species,
            dtype=torch.int32,
            device=DEVICE,
        ).unsqueeze(0)
        # [1, n_species]

        emb_sites = self.emd_site(idx_sites)
        # [bz, n_sites_pack, d_hid]
        emb_species = self.emd_species(idx_species)
        # [bz, n_species_pack, d_hid]

        if self.is_layernorm is True:
            emb_sites = self.layernorm(emb_sites)
            emb_species = self.layernorm(emb_species)

        emb_sites = emb_sites.detach().cpu().numpy()
        emb_species = emb_species.detach().cpu().numpy()

        return emb_sites, emb_species
