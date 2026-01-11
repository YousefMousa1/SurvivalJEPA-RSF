import argparse
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.configs import build_parser
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from src.encoder import Encoder
from src.torch_dataset import TorchDataset
from src.utils.encode_utils import encode_data


def build_args():
    parser = build_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--export_batch_size", type=int, default=256)
    parser.add_argument(
        "--use_target_encoder",
        action="store_true",
        help="If set, use target encoder weights. Defaults to context encoder.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
    dataset.load()

    export_args = Namespace(
        batch_size=args.export_batch_size,
        val_batch_size=-1,
        test_size_ratio=0.0,
        val_size_ratio=0.0,
        random_state=args.random_state,
        full_dataset_cuda=False,
        mock=False,
    )

    torch_dataset = TorchDataset(
        dataset=dataset,
        mode="train",
        kwargs=export_args,
        device=device,
        preprocessing=encode_data,
    )

    dataloader = DataLoader(
        dataset=torch_dataset,
        batch_size=args.export_batch_size,
        shuffle=False,
        num_workers=args.data_loader_nprocs,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    encoder = Encoder(
        idx_num_features=dataset.num_features,
        cardinalities=dataset.cardinalities,
        hidden_dim=args.model_dim_hidden,
        num_layers=args.model_num_layers,
        num_heads=args.model_num_heads,
        p_dropout=args.model_dropout_prob,
        layer_norm_eps=args.model_layer_norm_eps,
        gradient_clipping=args.exp_gradient_clipping,
        feature_type_embedding=args.model_feature_type_embedding,
        feature_index_embedding=args.model_feature_index_embedding,
        dim_feedforward=args.model_dim_feedforward,
        device=device,
        args=args,
    )

    state = torch.load(args.checkpoint, map_location=device)
    key = "target_encoder" if args.use_target_encoder else "context_encoder"
    encoder.load_state_dict(state[key])
    encoder.to(device)
    encoder.eval()

    embeddings = []
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device, non_blocking=True)
            z = encoder(batch)
            cls_embed = z[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embed)

    embeddings = np.concatenate(embeddings, axis=0)

    data = pd.read_csv(args.input_csv)
    out = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    out = pd.concat([data[["day", "status"]].reset_index(drop=True), out], axis=1)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote embeddings to {args.output_csv} with shape {out.shape}.")


if __name__ == "__main__":
    main()
