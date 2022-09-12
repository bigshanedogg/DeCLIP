import os
import torch
import torchvision
from setproctitle import setproctitle

setproctitle("hodonglee-clip-svo")

import ast
import PIL
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import prototype.solver.crash_on_ipy
from prototype.solver.clip_solver import ClsSolver as ClipSolver
from prototype.solver.slip_solver import ClsSolver as SlipSolver
from prototype.solver.filip_solver import ClsSolver as FilipSolver
from prototype.solver.declip_solver import ClsSolver as DeclipSolver
from prototype.solver.defilip_solver import ClsSolver as DefilipSolver

from utils import get_idx_string, build_transform


def main(
        architecture: str = "defilip",  # ["clip", "slip", "filip", "declip", "defilip"]
        image_backbone: str = "vit",
        batch_size: int = 128,
        src_filepath: str = "/home/bigshane/cloned/svo_probes/svo_probes.csv",
        image_dirpath: str = "/home/bigshane/cloned/svo_probes/images",
        output_dirpath: str = "./outputs/svo_default",
        extension: str = "jpg",
):
    print(f"# [info] architecture: {architecture}\timage_backbone: {image_backbone}")
    image_dirpath = Path(image_dirpath)
    output_dirpath = Path(output_dirpath)

    # load dataset
    df = pd.read_csv(src_filepath)
    df = df[df["verb_neg"]]

    # define transform & solver
    transform = build_transform()
    config_file_path = f"./experiments/{architecture}_experiments/yfcc15m/yfcc15m_{image_backbone}_{architecture}/config.yaml"

    solver = None
    if architecture == "clip":
        solver = ClipSolver(config_file_path)
    elif architecture == "slip":
        solver = SlipSolver(config_file_path)
    elif architecture == "filip":
        solver = FilipSolver(config_file_path)
    elif architecture == "declip":
        solver = DeclipSolver(config_file_path)
    elif architecture == "defilip":
        solver = DefilipSolver(config_file_path)

    # evaluation
    scores = np.empty((0, 2), dtype=np.float32)
    accs = []

    targets = df[["sentence", "pos_image_id", "neg_image_id"]].values.tolist()
    idxes = []
    batch = defaultdict(list)
    for row_idx, row in tqdm(enumerate(targets), initial=0, total=len(targets), desc="Compute metrics"):
        sentence, pos_image_id, neg_image_id = row
        pos_image_filepath = image_dirpath / f"{get_idx_string(pos_image_id)}.{extension}"
        neg_image_filepath = image_dirpath / f"{get_idx_string(neg_image_id)}.{extension}"
        if not pos_image_filepath.exists():
            # print(f"pos_image_filepath: {pos_image_filepath}")
            continue
        if not neg_image_filepath.exists():
            # print(f"neg_image_filepath: {neg_image_filepath}")
            continue

        try:
            pos_image = Image.open(pos_image_filepath)
            neg_image = Image.open(neg_image_filepath)
            if pos_image.mode != "RGB":
                pos_image = pos_image.convert("RGB")
            if neg_image.mode != "RGB":
                neg_image = neg_image.convert("RGB")

            if len(batch["pos_image"]) < 1 or len(batch["neg_image"]) < 1:
                batch["pos_image"] = torch.tensor([]).cuda()
                batch["neg_image"] = torch.tensor([]).cuda()

            pos_image = transform(pos_image).unsqueeze(0).cuda()
            batch["pos_image"] = torch.cat([batch["pos_image"], pos_image], dim=0)
            neg_image = transform(neg_image).unsqueeze(0).cuda()
            batch["neg_image"] = torch.cat([batch["neg_image"], neg_image], dim=0)
            batch["row_idx"].append(row_idx)
            batch["sentence"].append(sentence)

        except Exception as ex:
            print(f"# [warn] Invalid image - {ex}")

        if row_idx >= len(df) - 1 or len(batch["sentence"]) >= batch_size:
            # compute metric
            batch_outputs = _step(model=solver.model, batch=batch)
            scores = np.concatenate([scores, batch_outputs["scores"]], axis=0)
            accs.append(batch_outputs["acc"])
            idxes += batch["row_idx"]
            cur_batch_size = len(batch["sentence"])
            # print(f"row_idx: {row_idx}\tcur_batch_size: {cur_batch_size}\tcur_total: {len(scores)}")

            # initialize batch
            batch = defaultdict(list)

    # print & write outputs
    print(f"# [info] architecture: {architecture}\timage_backbone: {image_backbone}")
    print(f"total: {len(scores)}\tacc: {np.mean(accs)}")
    if not output_dirpath.exists():
        os.makedirs(output_dirpath, exist_ok=True)
    with open(output_dirpath / f"{architecture}_{image_backbone}_scores.txt", "w") as fp:
        for idx, score in zip(idxes, scores):
            row = str(idx) + "\t" + "\t".join([str(_score) for _score in score]) + "\n"
            fp.write(row)


def _step(model, batch):
    labels = torch.zeros(len(batch["sentence"])).cuda()

    # encode text & normalize
    text_embeddings = model.encode_text(batch["sentence"])
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # encode image & normalize
    pos_image_embeddings = model.encode_image(batch["pos_image"])
    pos_image_embeddings = pos_image_embeddings / pos_image_embeddings.norm(dim=-1, keepdim=True)
    neg_image_embeddings = model.encode_image(batch["neg_image"])
    neg_image_embeddings = neg_image_embeddings / neg_image_embeddings.norm(dim=-1, keepdim=True)

    # compute_metric
    pos_scores = torch.diagonal(text_embeddings @ pos_image_embeddings.T)
    neg_scores = torch.diagonal(text_embeddings @ neg_image_embeddings.T)
    scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores.unsqueeze(-1)], dim=-1)
    preds = torch.argmax(scores, dim=-1)
    acc = (labels == preds).sum() / labels.shape[0]
    return {
        "scores": scores.detach().cpu().numpy(),
        "acc": acc.item(),
    }


if __name__ == "__main__":
    main()