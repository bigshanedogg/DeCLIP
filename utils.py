import os
import torch
import torchvision
import ast
import PIL
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def get_idx_string(idx: int):
    idx_12 = f"{idx:0>12}"
    idx_str = f"{idx_12[:3]}/{idx_12[3:6]}/{idx_12[6:9]}/{idx_12[9:12]}"
    return idx_str


def open_image(image_dirpath: Path, image_id: int, extension: str = "jpg"):
    image_filepath = image_dirpath / f"{get_idx_string(image_id)}.{extension}"
    image = Image.open(image_filepath)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def build_transform():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def get_encoded_images(
        transform,
        model,
        image_dirpath: str,
        image_ids: List[int],
        extension: str = "jpg",
        verbose: bool = False,
):
    image_arrs = None
    mask = [False for _ in range(0, len(image_ids))]
    image_dirpath = Path(image_dirpath)
    for idx, image_id in enumerate(image_ids):
        image_arr = None
        try:
            image = open_image(image_dirpath, image_id, extension)
            image_arr = transform(image).unsqueeze(0).detach().cpu().numpy()
        except Exception as ex:
            if verbose:
                print(f"# [warn] Invalid image - {ex}")

        if image_arr is None:
            continue

        if model is not None:
            image_tensor = torch.tensor(image_arr).cuda()
            image_tensor = model.encode_image(image_tensor)
            image_arr = image_tensor.detach().cpu().numpy()

        if image_arrs is None:
            image_shape = tuple([len(image_ids)] + list(image_arr.shape[1:]))
            image_arrs = np.zeros(image_shape)

        image_arrs[idx] = image_arr
        mask[idx] = True

    return image_arrs, mask


def get_encoded_texts(model, texts: List[str]):
    text_embeddings = None
    for text in texts:
        text_embedding = model.encode_text([text])
        text_embedding = text_embedding.detach().cpu().numpy()
        if text_embeddings is None:
            text_embeddings = text_embedding
        else:
            text_embeddings = np.concatenate([text_embeddings, text_embedding], axis=0)
    return text_embeddings


def make_triplet_df(df, filter_invalid: bool = True):
    # melt df to triplet_df
    pos_df = df[["sentence", "pos_triplet", "pos_image_id"]]
    pos_df = pos_df.rename(columns={"pos_triplet": "triplet", "pos_image_id": "image_id"})
    neg_df = df[["sentence", "neg_triplet", "neg_image_id"]]
    neg_df.loc[:, "sentence"] = ["" for _ in range(len(neg_df))]
    neg_df = neg_df.rename(columns={"neg_triplet": "triplet", "neg_image_id": "image_id"})
    triplet_df = pd.concat([pos_df, neg_df], axis=0).drop_duplicates()

    # split triplet into (subj, verb, obj)
    subj, verb, obj, is_valid = list(), list(), list(), list()
    for triplet in triplet_df["triplet"]:
        _subj, _verb, _obj = "temp", "temp", "temp"
        _is_valid = True

        temp = triplet
        if "[" in triplet and "]" in triplet:
            triplet = ast.literal_eval(triplet)
            if len(triplet) == 1:
                triplet = triplet[0]
                _subj, _verb, _obj = triplet.split(",")
            else:
                _is_valid = False
        else:
            _subj, _verb, _obj = triplet.split(",")

        subj.append(_subj)
        verb.append(_verb)
        obj.append(_obj)
        is_valid.append(_is_valid)

    triplet_df["subj"] = subj
    triplet_df["verb"] = verb
    triplet_df["obj"] = obj
    triplet_df["is_valid"] = is_valid

    if filter_invalid:
        triplet_df = triplet_df[triplet_df["is_valid"]]
    return triplet_df


def make_key_dict(triplet_df, from_key: str = "subj", to_key: str = "verb"):
    key_dict = dict()
    for row_idx, row in enumerate(
            triplet_df[["image_id", "sentence", "triplet", "subj", "verb", "obj"]].values.tolist()):
        image_id, sentence, triplet, subj, verb, obj = row

        # define keys
        f_key = subj
        if from_key == "verb":
            f_key = verb
        elif from_key == "obj":
            f_key = obj
        t_key = verb
        if to_key == "subj":
            t_key = subj
        elif to_key == "obj":
            t_key = obj

        # initialize
        if f_key not in key_dict:
            key_dict[f_key] = set()
        key_dict[f_key].add(t_key)
    return key_dict


class Metric:
    @classmethod
    def recall_k(cls, scores: List[List[float]], labels: List[int], ks: List[int], inbatch_exclude: bool = False):
        """
        Args:
            scores: (batch_size, num_classes)
            labels: (batch_size, )
            ks: each k should be over than num_classes
        """
        output = dict()
        _predictions = np.argsort(-1 * scores, axis=-1)
        _labels = np.expand_dims(labels, axis=-1)

        if inbatch_exclude:
            _predictions_orig = np.argsort(-1 * scores, axis=-1)
            _labels_orig = labels.tolist()

            _predictions = []
            for i, _prediction in enumerate(_predictions_orig):
                _labels_to_exclude = _labels_orig[:i] + _labels_orig[i + 1:]
                _prediction = _prediction[~np.isin(_prediction, _labels_to_exclude)]
                _predictions.append(_prediction.tolist())
            _predictions = np.array(_predictions)

        denom, num_classes = _predictions.shape
        for k in ks:
            if k > num_classes:
                output[f"recall@{k}"] = -1
                continue

            predictions = _predictions[:, :k]
            labels = np.repeat(_labels, repeats=k, axis=1)
            nom = (predictions == labels).any(-1).sum()
            acc = nom / denom
            output[f"recall@{k}"] = acc
        return output

    @classmethod
    def rank(cls, scores: List[List[float]], labels: List[int], metric_only: bool = True,
             inbatch_exclude: bool = False):
        """
        Args:
            scores: (batch_size, num_classes)
            labels: (batch_size, )
        """
        output = dict()
        _predictions = np.argsort(-1 * scores, axis=-1)
        _labels = np.expand_dims(labels, axis=-1)

        if inbatch_exclude:
            _predictions_orig = np.argsort(-1 * scores, axis=-1)
            _labels_orig = labels.tolist()

            _predictions = []
            for i, _prediction in enumerate(_predictions_orig):
                _labels_to_exclude = _labels_orig[:i] + _labels_orig[i + 1:]
                _prediction = _prediction[~np.isin(_prediction, _labels_to_exclude)]
                _predictions.append(_prediction.tolist())
            _predictions = np.array(_predictions)

        rank = np.where(_predictions == _labels)[-1]
        normalized_rank = rank / _predictions.shape[-1]
        output["avg_rank"] = np.mean(normalized_rank)

        if not metric_only:
            output["rank"] = rank
            output["normalized_rank"] = normalized_rank
        return output