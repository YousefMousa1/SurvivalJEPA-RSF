import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.encoder import Encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune T-JEPA encoder with Cox loss.")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--horizon_years", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Encoder architecture (must match pretraining)
    parser.add_argument("--model_dim_hidden", type=int, default=64)
    parser.add_argument("--model_num_layers", type=int, default=4)
    parser.add_argument("--model_num_heads", type=int, default=8)
    parser.add_argument("--model_dim_feedforward", type=int, default=256)
    parser.add_argument("--model_dropout_prob", type=float, default=0.1)
    parser.add_argument("--model_layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--model_act_func", type=str, default="relu")
    parser.add_argument("--model_feature_type_embedding", action="store_true")
    parser.add_argument("--model_feature_index_embedding", action="store_true")
    parser.add_argument("--n_cls_tokens", type=int, default=1)
    return parser.parse_args()


class CoxDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.time = torch.tensor(time, dtype=torch.float32)
        self.event = torch.tensor(event, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]


def prepare_features(
    data: pd.DataFrame,
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
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

    return features.to_numpy(), num_features, cardinalities


def scale_numeric(train_X, val_X, num_features):
    scalers = {}
    for idx in num_features:
        scaler = MinMaxScaler()
        scaler.fit(train_X[:, [idx]])
        train_X[:, idx] = scaler.transform(train_X[:, [idx]]).reshape(-1)
        val_X[:, idx] = scaler.transform(val_X[:, [idx]]).reshape(-1)
        scalers[idx] = scaler
    return train_X, val_X


def cox_partial_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.sum((risk - log_cumsum) * event) / (event.sum() + 1e-8)
    return loss


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data = pd.read_csv(args.data_csv)
    data = data.dropna(subset=["day", "status"])

    time_years = data["day"].astype(float).to_numpy() / 365.25
    status = data["status"].astype(str).str.lower().to_numpy()
    horizon = float(args.horizon_years)
    time = np.minimum(time_years, horizon)
    event = ((status == "dead") & (time_years <= horizon)).astype(np.float32)

    X, num_features, cardinalities = prepare_features(data)

    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_ratio, random_state=args.seed, shuffle=True
    )
    X_train = X[train_idx].copy()
    X_val = X[val_idx].copy()
    X_train, X_val = scale_numeric(X_train, X_val, num_features)

    train_ds = CoxDataset(X_train, time[train_idx], event[train_idx])
    val_ds = CoxDataset(X_val, time[val_idx], event[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        idx_num_features=num_features,
        cardinalities=cardinalities,
        hidden_dim=args.model_dim_hidden,
        num_layers=args.model_num_layers,
        num_heads=args.model_num_heads,
        p_dropout=args.model_dropout_prob,
        layer_norm_eps=args.model_layer_norm_eps,
        gradient_clipping=0.0,
        feature_type_embedding=args.model_feature_type_embedding,
        feature_index_embedding=args.model_feature_index_embedding,
        dim_feedforward=args.model_dim_feedforward,
        device=device,
        args=args,
    )
    state = torch.load(args.checkpoint, map_location=device)
    key = "target_encoder" if "target_encoder" in state else "context_encoder"
    encoder.load_state_dict(state[key])
    encoder.to(device)

    head = nn.Linear(args.model_dim_hidden, 1).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        train_losses = []
        for batch_X, batch_time, batch_event in train_loader:
            batch_X = batch_X.to(device)
            batch_time = batch_time.to(device)
            batch_event = batch_event.to(device)

            z = encoder(batch_X)
            cls_embed = z[:, 0, :]
            risk = head(cls_embed).squeeze(-1)
            loss = cox_partial_loss(risk, batch_time, batch_event)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        encoder.eval()
        head.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_time, batch_event in val_loader:
                batch_X = batch_X.to(device)
                batch_time = batch_time.to(device)
                batch_event = batch_event.to(device)
                z = encoder(batch_X)
                cls_embed = z[:, 0, :]
                risk = head(cls_embed).squeeze(-1)
                loss = cox_partial_loss(risk, batch_time, batch_event)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"epoch {epoch+1}/{args.epochs} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {
                "context_encoder": encoder.state_dict(),
                "cox_head": head.state_dict(),
                "val_loss": best_val,
                "horizon_years": horizon,
                "seed": args.seed,
            }

    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    torch.save(best_state, args.output_checkpoint)
    print(f"Saved fine-tuned checkpoint to {args.output_checkpoint}")


if __name__ == "__main__":
    main()
