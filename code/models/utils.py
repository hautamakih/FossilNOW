import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame


class Paths:
    def __init__(self, root: str = "data_processed") -> None:
        self.encoding_sites = Path(root) / "encoding" / "ordinal_enc_site.json"
        self.encoding_genera = Path(root) / "encoding" / "ordinal_enc_genus.json"

        self.embd_sites_prob_false = Path(root) / "mf" / "prob_False" / "emb_sites.npy"
        self.embd_sites_prob_true = Path(root) / "mf" / "prob_True" / "emb_sites.npy"
        self.embd_genera_prob_false = (
            Path(root) / "mf" / "prob_False" / "emb_genera.npy"
        )
        self.embd_genera_prob_true = Path(root) / "mf" / "prob_True" / "emb_genera.npy"

        self.weight_mf_prob_false = Path(root) / "mf" / "prob_False" / "model.pt"
        self.weight_mf_prob_true = Path(root) / "mf" / "prob_True" / "model.pt"
        self.weight_knn_prob_false = Path(root) / "knn" / "prob_False"
        self.weight_knn_prob_true = Path(root) / "knn" / "prob_True"

        self._create_dir()

    def _create_dir(self):
        for path in [
            self.encoding_sites,
            self.encoding_genera,
            self.weight_mf_prob_false,
            self.weight_mf_prob_true,
            self.weight_knn_prob_false,
            self.weight_knn_prob_true,
            self.embd_sites_prob_false,
            self.embd_sites_prob_true,
            self.embd_genera_prob_false,
            self.embd_genera_prob_true,
        ]:
            path.parent.mkdir(exist_ok=True, parents=True)


paths = Paths()


class CategoryDict:
    def __init__(self) -> None:
        self.dict_name2id = {}
        self.dict_id2name = {}

    def names2ids(self, names) -> list:
        if isinstance(names, str):
            names = [names]

        ids = []
        for name in names:
            assert name in self.dict_name2id

            ids.append(self.dict_name2id[name])

        return ids

    def ids2names(self, ids):
        if isinstance(ids, str | int):
            ids = [ids]

        names = []
        for idx in ids:
            assert idx in self.dict_id2name

            names.append(self.dict_id2name[idx])

        return names

    def save_dict(self, path: str | Path):
        with open(path, "w+") as f:
            json.dump(self.dict_id2name, f, indent=2)

    def size(self):
        return len(self.dict_id2name)

    @classmethod
    def from_file(cls, path: str | Path):
        assert Path(path).exists()

        with open(path) as f:
            d = json.load(f)

        obj = CategoryDict()
        for idx, name in d.items():
            obj.dict_id2name[int(idx)] = name
            obj.dict_name2id[name] = int(idx)

        return obj

    @classmethod
    def from_list(cls, names: list):
        obj = CategoryDict()

        for idx, name in enumerate(names):
            obj.dict_id2name[idx] = name
            obj.dict_name2id[name] = idx

        return obj


def conv_dataset_patch2df(patches: list) -> pd.DataFrame:
    records = []

    for x in patches:
        for i_site, site in enumerate(x["site"]):
            for i_gen, genus in enumerate(x["genus"]):
                records.append(
                    {
                        "site": site,
                        "genus": genus,
                        "occurence": x["occurence"][i_site, i_gen],
                    }
                )

    df = pd.DataFrame.from_records(records)

    return df


def _iterate_pack(values: list, n: int):
    # assert len(values) % n == 0

    for i in range(0, len(values), n):
        yield values[i : i + n]


def split_traintest(
    df: DataFrame,
    is_packed: bool = False,
    is_encoded: bool = False,
    is_seed: bool = True,
    ratio_traintest: float = 0.8,
    num_sites_per_pack: int = 2,
    num_genera_per_pack: int = 4,
) -> tuple[DataFrame, DataFrame]:
    """Process and split 'df' into 2 train and test dataframes. Data is separated into packs.
    Each pack contains the occurence of 'num_genera_per_pack' genera in 'num_sites_per_pack' sites.\\
    Example:
    
    path = "../data/data_occ.csv" \\
    df = pd.read_csv(path, delimiter='\t') \\
    df_train, df_test = utils.split_traintest(df)

    Args:
        df (DataFrame): input dataframe
        is_packed (bool, optional): Whether forming packing data (this is useful for Matrix Factorization only). Defaults to False.
        is_encoded (bool, optional): Whether the site and genus are encoded. Defaults to False.
        is_seed (bool, optional): Whether seed everything to guarantee same train/test. Defaults to True.
        ratio_traintest (float, optional): Train/test data ratio. Defaults to 0.8.
        num_sites_per_pack (int, optional): number of sites in each pack. Defaults to 2.
        num_genera_per_pack (int, optional): number of genera in each pack. Defaults to 4.

    Returns:
        tuple[DataFrame, DataFrame]: train and test dataframe
    """

    if is_seed is True:
        random.seed(1)
        np.random.seed(1)

    cols_redundant = [
        "LOC",
        "LIDNUM",
        # 'NAME',
        "COUNTRY",
        "MID_AGE",
        "MAX_AGE",
        "MIN_AGE",
        "LATSTR",
        "LONGSTR",
        "n_gen",
    ]
    cols_redundant_real = []
    for col in cols_redundant:
        if col in df:
            cols_redundant_real.append(col)
    df = df.drop(columns=cols_redundant_real)

    # Check if index is already set
    if df.index.name != "SITE_NAME" or df.index.name != "NAME":
        for idx_name in ["NAME", "SITE_NAME"]:
            if idx_name in df:
                df = df.set_index(idx_name)

                break

    # Prepare site and genus encoder
    list_sites = df.index
    list_genera = df.columns
    enc_genus = CategoryDict.from_list(list_genera)
    enc_site = CategoryDict.from_list(list_sites)

    # print(list_sites)

    # Start creating data
    data = []
    for sites, genera in itertools.product(
        _iterate_pack(list_sites, num_sites_per_pack),
        _iterate_pack(list_genera, num_genera_per_pack),
    ):
        occurence = df.loc[sites, genera].to_numpy().astype(np.float32)

        if is_encoded is True:
            sites_encoded = enc_site.names2ids(sites)
            genera_encoded = enc_genus.names2ids(genera)
        else:
            sites_encoded = sites
            genera_encoded = genera

        if (
            len(sites_encoded) == num_sites_per_pack
            and len(genera_encoded) == num_genera_per_pack
        ):
            data.append(
                {
                    "occurence": occurence,
                    "site": sites_encoded,
                    "genus": genera_encoded,
                }
            )

    enc_genus.save_dict(paths.encoding_genera)
    enc_site.save_dict(paths.encoding_sites)

    # Split train/test
    random.shuffle(data)
    len_train = int(len(data) * ratio_traintest)
    data_train, data_test = data[:len_train], data[len_train:]

    # Unpack or pack data
    # For Matrix Factorization, it's better if the data is pack.
    # But for other algorithms, unpacking data is more suitable
    if is_packed is False:

        def _unpack(data: list):
            list_out = []
            for r in data:
                occ = r["occurence"].flatten()
                i = 0
                for site in r["site"]:
                    for genus in r["genus"]:
                        list_out.append(
                            {"site": site, "genus": genus, "occurence": occ[i]}
                        )
                        i += 1
            return list_out

        list_train_unpacked = _unpack(data_train)
        list_test_unpacked = _unpack(data_test)

        df_train = pd.DataFrame.from_records(list_train_unpacked)
        df_test = pd.DataFrame.from_records(list_test_unpacked)

    else:
        df_train = pd.DataFrame.from_records(data_train)
        df_test = pd.DataFrame.from_records(data_test)

    return df_train, df_test
