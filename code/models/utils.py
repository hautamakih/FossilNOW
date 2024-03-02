import json
from pathlib import Path


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

    def save_dict(self, path: str):
        with open(path, "w+") as f:
            json.dump(self.dict_id2name, f, indent=2)

    def size(self):
        return len(self.dict_id2name)

    @classmethod
    def from_file(cls, path: str):
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
