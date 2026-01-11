import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE


class MetabricTable6(BaseDataset):
    def __init__(self, args):
        super(MetabricTable6, self).__init__(args)
        self.is_data_loaded = False
        self.tmp_file_names = ["metabric_rsf_table6.csv"]
        self.name = "metabric_table6"
        self.args = args
        self.task_type = TASK_TYPE.BINARY_CLASS

    def load(self):
        if self.is_data_loaded:
            return

        path = os.path.join(self.args.data_path, self.tmp_file_names[0])
        data = pd.read_csv(path)
        data = data.dropna(subset=["day", "status"])

        status_map = {"dead": 1, "alive": 0}
        self.y = data["status"].astype(str).str.lower().map(status_map).to_numpy()

        features = data.drop(columns=["day", "status"]).copy()

        for col in features.columns:
            if features[col].dtype == object:
                features[col] = features[col].fillna("unknown").astype(str)
            else:
                features[col] = pd.to_numeric(features[col], errors="coerce")
                features[col] = features[col].fillna(features[col].median())

        cat_features = []
        num_features = []
        cardinalities = []

        for idx, col in enumerate(features.columns):
            if features[col].dtype == object:
                encoder = LabelEncoder()
                features[col] = encoder.fit_transform(features[col])
                cat_features.append(idx)
                cardinalities.append((idx, len(encoder.classes_)))
            else:
                num_features.append(idx)

        self.X = features.to_numpy()
        self.N, self.D = self.X.shape
        self.cat_features = cat_features
        self.num_features = num_features
        self.cardinalities = cardinalities
        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}
        self.is_data_loaded = True
