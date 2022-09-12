import os
import torch
import torchvision
from setproctitle import setproctitle

setproctitle("hodonglee-clip-svo-ft")

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

from utils import get_idx_string, build_transform, get_encoded_images, get_encoded_texts, make_triplet_df, \
    make_key_dict, Metric


def main(
        architecture: str = "clip",  # ["clip", "slip", "filip", "declip"]
        image_backbone: str = "vit",
        src_filepath: str = "/home/bigshane/cloned/svo_probes/svo_probes.csv",
        image_dirpath: str = "/home/bigshane/cloned/svo_probes/images",
        output_dirpath: str = "./outputs/svo_ft",
        extension: str = "jpg",
        from_key: str = "subj",
        to_key: str = "verb",
        ks: List[int] = [1, 2, 5, 10, 20, 100],
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

    # preprocess data
    triplet_df = make_triplet_df(df, filter_invalid=True)
    triplet_df = triplet_df[triplet_df["sentence"] != ""]
    key_dict = make_key_dict(triplet_df, from_key=from_key, to_key=to_key)

    # evaluation
    total_outputs = {"from_key": from_key, "to_key": to_key}
    f_key_outputs = defaultdict(list)
    for idx, (f_key, t_key_set) in tqdm(enumerate(key_dict.items()), initial=0, total=len(key_dict),
                                        desc=f"Evaluate svo_{from_key}-{to_key}"):
        if idx > 3:
            continue
        subset_df = triplet_df[triplet_df[from_key] == f_key]
        subset_image_ids = subset_df["image_id"].tolist()
        subset_sentences = subset_df["sentence"].tolist()

        # encode image
        subset_image_embeddings, subset_mask = get_encoded_images(
            transform=transform,
            model=solver.model,
            image_dirpath=image_dirpath,
            image_ids=subset_image_ids,
            extension=extension,
        )
        if np.sum(subset_mask) < 1: continue
        subset_image_embeddings = subset_image_embeddings[subset_mask]
        subset_image_embeddings = subset_image_embeddings / np.linalg.norm(subset_image_embeddings, axis=-1,
                                                                           keepdims=True)

        # encode text
        subset_text_embeddings = get_encoded_texts(
            model=solver.model,
            texts=subset_sentences,
        )
        subset_text_embeddings = subset_text_embeddings[subset_mask]
        subset_text_embeddings = subset_text_embeddings / np.linalg.norm(subset_text_embeddings, axis=-1, keepdims=True)

        subset_scores = subset_text_embeddings @ subset_image_embeddings.T

        t_key_outputs = defaultdict(list)
        for t_key in t_key_set:
            t_mask = np.array((subset_df[to_key] == t_key).astype(bool).tolist())
            t_mask = t_mask[subset_mask]
            labels = np.where(t_mask)[0]
            if len(labels) < 1:
                continue
            scores = subset_scores[labels]

            # update metrics
            outputs = {"size": scores.shape[-1]}
            _outputs = Metric.recall_k(scores, labels, ks=ks, inbatch_exclude=True)
            outputs.update(_outputs)
            _outputs = Metric.rank(scores, labels, inbatch_exclude=True)
            outputs.update(_outputs)

            for k, v in outputs.items():
                t_key_outputs[k].append(v)
            t_key_outputs["from_key"] = f_key

        # aggregate t_key metrics into f_key metrics
        for k, v in t_key_outputs.items():
            if k in ["from_key"]:
                t_key_outputs[k] = v
            else:
                t_key_outputs[k] = np.mean(v)
        for k, v in t_key_outputs.items():
            f_key_outputs[k].append(v)

    # aggregate f_key metrics into scalars
    for k, v in f_key_outputs.items():
        if k in ["from_key"]:
            continue
        elif k in ["size"]:
            total_outputs[k] = np.sum(v)
        else:
            total_outputs[k] = np.mean(v)

    # print & write outputs
    print(f"# [info] architecture: {architecture}\timage_backbone: {image_backbone}")
    for k, v in total_outputs.items():
        print(f"{k}: {v}")
    if not output_dirpath.exists():
        os.makedirs(output_dirpath, exist_ok=True)
    with open(output_dirpath / f"{architecture}_{image_backbone}_scores.txt", "w") as fp:
        f_keys = f_key_outputs.pop("from_key")
        header = list(f_key_outputs.keys())
        fp.write("\t".join(header) + "\n")
        for i in range(0, len(f_keys)):
            row_str = f"{f_keys[i]}"
            for _header in header:
                if _header not in f_key_outputs:
                    continue
                v = f_key_outputs[_header][i]
                row_str += f"{v}\t"
            row_str += "\n"
            fp.write(row_str)


if __name__ == "__main__":
    main()