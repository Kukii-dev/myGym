"""
Training script for the Reasoner module.

Usage:
    python -m myGym.reasoner.train --data path/to/dataset.json [--epochs 50] [--lr 1e-4]

Metrics:
    - Object grounding: Accuracy and F1 score
    - Action sequence:  Token accuracy and Levenshtein distance
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from myGym.reasoner.model import ReasonerModel
from myGym.reasoner.dataset import ReasonerDataset, Vocabulary, collate_fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def levenshtein_distance(pred: List, target: List) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    n, m = len(pred), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def compute_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Macro F1 over object classes."""
    f1_scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if (targets == c).any():
            f1_scores.append(f1.item())
    return sum(f1_scores) / max(len(f1_scores), 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ReasonerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    action_pad_id: int,
    obj_loss_weight: float = 1.0,
    action_loss_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_obj_loss = 0.0
    total_act_loss = 0.0
    total_obj_correct = 0
    total_samples = 0

    for batch in loader:
        pose = batch["pose"].to(device)
        objects = batch["objects"].to(device)
        obj_cat = batch["obj_cat"].to(device)
        obj_mask = batch["obj_mask"].to(device)
        lang_tokens = batch["lang_tokens"].to(device)
        lang_mask = batch["lang_mask"].to(device)
        action_input = batch["action_input"].to(device)
        action_mask = batch["action_mask"].to(device)
        target_idx = batch["target_idx"].to(device)

        # Action targets: shift left by 1
        action_target = action_input.clone()
        action_target[:, :-1] = action_input[:, 1:]
        action_target[:, -1] = action_pad_id

        out = model(
            pose=pose,
            objects=objects,
            obj_cat=obj_cat,
            lang_tokens=lang_tokens,
            action_tokens=action_input,
            obj_mask=obj_mask,
            lang_mask=lang_mask,
            action_mask=action_mask,
        )

        # Object grounding loss
        obj_loss = F.cross_entropy(out["obj_logits"], target_idx)

        # Action sequence loss (ignore padding)
        act_logits = out["action_logits"].reshape(-1, out["action_logits"].size(-1))
        act_targets = action_target.reshape(-1)
        action_loss = F.cross_entropy(act_logits, act_targets, ignore_index=action_pad_id)

        loss = obj_loss_weight * obj_loss + action_loss_weight * action_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = pose.size(0)
        total_samples += B
        total_obj_loss += obj_loss.item() * B
        total_act_loss += action_loss.item() * B
        total_obj_correct += (out["obj_logits"].argmax(dim=-1) == target_idx).sum().item()

    return {
        "obj_loss": total_obj_loss / total_samples,
        "action_loss": total_act_loss / total_samples,
        "obj_accuracy": total_obj_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: ReasonerModel,
    loader: DataLoader,
    device: torch.device,
    action_vocab: Vocabulary,
    action_pad_id: int,
) -> Dict[str, float]:
    model.eval()
    total_obj_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    total_lev = 0.0
    total_act_seqs = 0
    num_objects = 0

    for batch in loader:
        pose = batch["pose"].to(device)
        objects = batch["objects"].to(device)
        obj_cat = batch["obj_cat"].to(device)
        obj_mask = batch["obj_mask"].to(device)
        lang_tokens = batch["lang_tokens"].to(device)
        lang_mask = batch["lang_mask"].to(device)
        action_input = batch["action_input"].to(device)
        target_idx = batch["target_idx"].to(device)

        B = pose.size(0)
        total_samples += B

        result = model.predict(
            pose=pose,
            objects=objects,
            obj_cat=obj_cat,
            lang_tokens=lang_tokens,
            obj_mask=obj_mask,
            lang_mask=lang_mask,
            start_token_id=action_vocab.start_id,
            end_token_id=action_vocab.end_id,
        )

        obj_preds = result["obj_idx"]
        total_obj_correct += (obj_preds == target_idx).sum().item()
        all_preds.append(obj_preds.cpu())
        all_targets.append(target_idx.cpu())
        num_objects = max(num_objects, objects.size(1))

        # Levenshtein distance on action sequences
        for i in range(B):
            # Ground truth: strip START, END, PAD
            gt = action_input[i].cpu().tolist()
            gt = [t for t in gt if t not in (action_vocab.pad_id, action_vocab.start_id, action_vocab.end_id)]
            # Predicted: strip START, END
            pred = result["action_seq"][i].cpu().tolist()
            pred = [t for t in pred if t not in (action_vocab.pad_id, action_vocab.start_id, action_vocab.end_id)]
            total_lev += levenshtein_distance(pred, gt)
            total_act_seqs += 1

    all_preds_t = torch.cat(all_preds)
    all_targets_t = torch.cat(all_targets)

    return {
        "obj_accuracy": total_obj_correct / total_samples,
        "obj_f1": compute_f1(all_preds_t, all_targets_t, num_objects),
        "action_levenshtein": total_lev / max(total_act_seqs, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the Reasoner module")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--cat_embed_dim", type=int, default=None,
                        help="Categorical embedding dim (defaults to d_model for weight sharing)")
    parser.add_argument("--num_joints", type=int, default=18)
    parser.add_argument("--max_objects", type=int, default=16)
    parser.add_argument("--max_lang_tokens", type=int, default=32)
    parser.add_argument("--max_action_len", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--save_dir", type=str, default="reasoner_checkpoints")
    parser.add_argument("--obj_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Dataset ---
    dataset = ReasonerDataset(
        json_path=args.data,
        max_objects=args.max_objects,
        max_lang_tokens=args.max_lang_tokens,
        max_action_len=args.max_action_len,
        num_joints=args.num_joints,
    )
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Dataset: {len(dataset)} samples ({train_size} train / {val_size} val)")
    print(f"Language vocab: {len(dataset.lang_vocab)} tokens")
    print(f"Action vocab:   {len(dataset.action_vocab)} tokens")

    # --- Model ---
    cat_embed_dim = args.cat_embed_dim if args.cat_embed_dim is not None else args.d_model
    model = ReasonerModel(
        num_joints=args.num_joints,
        lang_vocab_size=len(dataset.lang_vocab),
        cat_embed_dim=cat_embed_dim,
        action_vocab_size=len(dataset.action_vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_objects=args.max_objects,
        max_action_len=args.max_action_len,
        max_seq_len=args.max_lang_tokens,
        pad_token_id=dataset.action_vocab.pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training ---
    os.makedirs(args.save_dir, exist_ok=True)
    best_obj_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            action_pad_id=dataset.action_vocab.pad_id,
            obj_loss_weight=args.obj_loss_weight,
            action_loss_weight=args.action_loss_weight,
        )
        val_metrics = evaluate(
            model, val_loader, device,
            action_vocab=dataset.action_vocab,
            action_pad_id=dataset.action_vocab.pad_id,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_obj_loss={train_metrics['obj_loss']:.4f}  "
            f"train_act_loss={train_metrics['action_loss']:.4f}  "
            f"train_obj_acc={train_metrics['obj_accuracy']:.3f}  |  "
            f"val_obj_acc={val_metrics['obj_accuracy']:.3f}  "
            f"val_f1={val_metrics['obj_f1']:.3f}  "
            f"val_lev={val_metrics['action_levenshtein']:.2f}"
        )

        if val_metrics["obj_accuracy"] > best_obj_acc:
            best_obj_acc = val_metrics["obj_accuracy"]
            ckpt_path = os.path.join(args.save_dir, "best_reasoner.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": vars(args),
                "lang_vocab": dataset.lang_vocab.token2id,
                "action_vocab": dataset.action_vocab.token2id,
            }, ckpt_path)
            print(f"  -> Saved best model (obj_acc={best_obj_acc:.3f})")

    # Save final model
    final_path = os.path.join(args.save_dir, "final_reasoner.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "lang_vocab": dataset.lang_vocab.token2id,
        "action_vocab": dataset.action_vocab.token2id,
    }, final_path)
    print(f"Training complete. Best val obj_accuracy: {best_obj_acc:.3f}")


if __name__ == "__main__":
    main()
