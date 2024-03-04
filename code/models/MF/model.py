import torch
import torch.nn as torch_nn
from torch import Tensor

DEVICE = "mps"
EPS = 1e-6


class MF(torch_nn.Module):
    def __init__(
        self,
        n_sites: int,
        n_species: int,
        d_hid: int = 64,
        prob_output: bool = False,
    ) -> None:
        super().__init__()

        self.prob_output = prob_output

        self.emd_site = torch_nn.Embedding(n_sites, d_hid)
        self.emd_species = torch_nn.Embedding(n_species, d_hid)
        self.batchnorm = torch_nn.BatchNorm2d(1)
        self.layernorm = torch_nn.LayerNorm(d_hid)

        self.act_prob = torch_nn.Sigmoid()

    def forward(self, idx_sites: Tensor, idx_species: Tensor):
        embedding_sites = self.emd_site(idx_sites)
        # [bz, n_sites_pack, d_hid]
        embedding_species = self.emd_species(idx_species)
        # [bz, n_species_pack, d_hid]

        emb_sites = self.layernorm(embedding_sites)
        emb_species = self.layernorm(embedding_species)

        emb_species_T = torch.permute(emb_species, (0, 2, 1))
        # [bz, d_hid, n_species_pack]

        occurence_pred = torch.bmm(emb_sites, emb_species_T)
        # [bz, n_sites_pack, n_species_pack]

        if self.prob_output is True:
            occurence_pred = self.act_prob(occurence_pred)

        return occurence_pred
