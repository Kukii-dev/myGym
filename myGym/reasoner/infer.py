"""
Inference script for the Reasoner module.

Loads a trained checkpoint and runs predictions on samples from a dataset,
printing the instruction, objects, ground-truth target, and model prediction.

Usage:
    python -m myGym.reasoner.infer \
        --checkpoint reasoner_checkpoints/best_reasoner.pt \
        --data myGym/reasoner/dataset.json \
        --n 20 \
        --split val        # 'val', 'train', or 'all'

Options:
    --n         Number of samples to evaluate (default 20, 0 = all)
    --split     Which split to sample from (default: val)
    --seed      Random seed for sample selection (default: 0)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from myGym.reasoner.dataset import ReasonerDataset, Vocabulary, collate_fn
from myGym.reasoner.model import ReasonerModel


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt["config"]

    # Reconstruct vocabularies from checkpoint
    lang_vocab   = Vocabulary()
    lang_vocab.token2id = ckpt["lang_vocab"]
    lang_vocab.id2token = {i: t for t, i in lang_vocab.token2id.items()}

    action_vocab = Vocabulary()
    action_vocab.token2id = ckpt["action_vocab"]
    action_vocab.id2token = {i: t for t, i in action_vocab.token2id.items()}

    cat_embed_dim = cfg.get("cat_embed_dim") or cfg.get("d_model", 128)

    model = ReasonerModel(
        num_joints       = cfg["num_joints"],
        lang_vocab_size  = len(lang_vocab),
        cat_embed_dim    = cat_embed_dim,
        action_vocab_size= len(action_vocab),
        d_model          = cfg["d_model"],
        n_heads          = cfg["n_heads"],
        max_objects      = cfg["max_objects"],
        max_action_len   = cfg["max_action_len"],
        max_seq_len      = cfg["max_lang_tokens"],
        pad_token_id     = action_vocab.pad_id,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, lang_vocab, action_vocab, cfg


def decode_action_seq(seq_ids, action_vocab):
    tokens = action_vocab.decode(seq_ids)
    return [t for t in tokens if t not in (
        Vocabulary.PAD, Vocabulary.START, Vocabulary.END
    )]


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = load_checkpoint(args.checkpoint)
    model, lang_vocab, action_vocab, cfg = build_model_from_ckpt(ckpt, device)

    val_metrics = ckpt.get("val_metrics", {})
    print(f"Checkpoint epoch : {ckpt.get('epoch', '?')}")
    if val_metrics:
        print(f"Saved val metrics: obj_acc={val_metrics.get('obj_accuracy', '?'):.3f}  "
              f"lev={val_metrics.get('action_levenshtein', '?'):.2f}")
    print()

    # Load dataset with vocabularies from checkpoint
    dataset = ReasonerDataset(
        json_path      = args.data,
        lang_vocab     = lang_vocab,
        action_vocab   = action_vocab,
        max_objects    = cfg["max_objects"],
        max_lang_tokens= cfg["max_lang_tokens"],
        max_action_len = cfg["max_action_len"],
        num_joints     = cfg["num_joints"],
    )

    # Split into train/val the same way training did
    val_size   = max(1, int(len(dataset) * cfg.get("val_split", 0.15)))
    train_size = len(dataset) - val_size
    rng        = torch.Generator().manual_seed(0)  # same seed as training
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=rng)

    if args.split == "val":
        subset = val_ds
    elif args.split == "train":
        subset = train_ds
    else:
        subset = dataset

    # Sample n items
    rnd = random.Random(args.seed)
    indices = list(range(len(subset)))
    rnd.shuffle(indices)
    n = min(args.n, len(indices)) if args.n > 0 else len(indices)
    chosen = Subset(subset, indices[:n])

    loader = DataLoader(chosen, batch_size=min(n, 32), collate_fn=collate_fn)

    correct_obj = 0
    total       = 0
    total_lev   = 0.0

    print(f"{'─'*72}")
    print(f"Split: {args.split}  |  Samples: {n}")
    print(f"{'─'*72}")

    sample_idx = 0
    for batch in loader:
        pose        = batch["pose"].to(device)
        objects     = batch["objects"].to(device)
        obj_cat     = batch["obj_cat"].to(device)
        obj_mask    = batch["obj_mask"].to(device)
        lang_tokens = batch["lang_tokens"].to(device)
        lang_mask   = batch["lang_mask"].to(device)
        target_idx  = batch["target_idx"]

        with torch.no_grad():
            result = model.predict(
                pose=pose, objects=objects, obj_cat=obj_cat,
                lang_tokens=lang_tokens, obj_mask=obj_mask, lang_mask=lang_mask,
                start_token_id=action_vocab.start_id,
                end_token_id=action_vocab.end_id,
            )

        pred_obj = result["obj_idx"].cpu()
        pred_seq = result["action_seq"].cpu()

        for i in range(pose.size(0)):
            raw   = dataset.raw_data[chosen.indices[sample_idx] if hasattr(chosen, 'indices') else sample_idx]
            instr = raw["instruction"]
            objs  = raw["objects"]
            gt    = target_idx[i].item()
            pred  = pred_obj[i].item()
            ok    = pred == gt

            correct_obj += int(ok)
            total       += 1

            # Decode action sequence
            seq_ids  = pred_seq[i].tolist()
            pred_act = decode_action_seq(seq_ids, action_vocab)
            gt_act   = raw["action_sequence"]

            # Format object list
            obj_strs = [f"{o['color']} {o['shape']}" for o in objs]
            obj_list = "  ".join(
                f"[{j}]{s}{'*' if j == gt else ''}"
                for j, s in enumerate(obj_strs)
            )

            status = "✓" if ok else "✗"
            print(f"{status}  \"{instr}\"")
            print(f"   Objects   : {obj_list}  (* = ground truth)")
            print(f"   Predicted : [{pred}] {obj_strs[pred] if pred < len(obj_strs) else '?'}"
                  f"  →  action: {pred_act}  (gt: {gt_act})")
            print()

            sample_idx += 1

    print(f"{'─'*72}")
    print(f"Object accuracy : {correct_obj}/{total} = {correct_obj/total:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--n",    type=int, default=20, help="0 = all")
    parser.add_argument("--split", default="val", choices=["val", "train", "all"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
