"""
Dataset and vocabulary utilities for the Reasoner module.

JSON dataset schema (list of samples):
```json
[
  {
    "pose": [[x,y,z], ...],              // K joints, 3 coords each
    "objects": [                          // N objects in the scene
      {
        "position": [x, y, z],
        "orientation": [roll, pitch, yaw],
        "size": "small",
        "color": "red",
        "shape": "cube",
        "texture": "smooth"
      }, ...
    ],
    "instruction": "pick up the red cube",
    "target_object_idx": 0,              // index into `objects`
    "action_sequence": ["A", "G", "M", "D", "W"]
  }, ...
]
```
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

class Vocabulary:
    """Simple token <-> id mapping with special tokens."""

    PAD = "<PAD>"
    START = "<START>"
    END = "<END>"
    UNK = "<UNK>"
    SPECIAL_TOKENS = [PAD, START, END, UNK]

    def __init__(self, tokens: Optional[List[str]] = None):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        for tok in self.SPECIAL_TOKENS:
            self._add(tok)
        if tokens:
            for tok in tokens:
                self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def encode(self, tokens: List[str]) -> List[int]:
        unk_id = self.token2id[self.UNK]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, self.UNK) for i in ids]

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD]

    @property
    def start_id(self) -> int:
        return self.token2id[self.START]

    @property
    def end_id(self) -> int:
        return self.token2id[self.END]

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.token2id, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls()
        with open(path) as f:
            v.token2id = json.load(f)
        v.id2token = {i: t for t, i in v.token2id.items()}
        return v


# ---------------------------------------------------------------------------
# Categorical property embedder
# ---------------------------------------------------------------------------

class PropertyEmbedder:
    """
    Maps categorical object properties (size, color, shape, texture) to
    fixed-length vectors via simple lookup tables built from training data.
    """

    PROPERTIES = ["size", "color", "shape", "texture"]

    def __init__(self):
        self.vocabs: Dict[str, Vocabulary] = {
            p: Vocabulary() for p in self.PROPERTIES
        }

    def fit(self, objects_list: List[List[Dict]]):
        for objects in objects_list:
            for obj in objects:
                for prop in self.PROPERTIES:
                    val = obj.get(prop, "unknown")
                    self.vocabs[prop]._add(val)

    def encode_object(self, obj: Dict) -> List[float]:
        """Returns [x,y,z, roll,pitch,yaw, size_id, color_id, shape_id, texture_id]."""
        pos = obj.get("position", [0.0, 0.0, 0.0])
        ori = obj.get("orientation", [0.0, 0.0, 0.0])
        cats = []
        for prop in self.PROPERTIES:
            val = obj.get(prop, "unknown")
            cats.append(float(self.vocabs[prop].token2id.get(val, 0)))
        return pos + ori + cats

    def save(self, path: str):
        data = {p: v.token2id for p, v in self.vocabs.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "PropertyEmbedder":
        pe = cls()
        with open(path) as f:
            data = json.load(f)
        for prop, mapping in data.items():
            pe.vocabs[prop].token2id = mapping
            pe.vocabs[prop].id2token = {i: t for t, i in mapping.items()}
        return pe


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReasonerDataset(Dataset):
    """
    PyTorch dataset that loads the reasoner JSON file.

    Args:
        json_path:        path to the JSON dataset
        lang_vocab:       Vocabulary for language tokens (built if None)
        action_vocab:     Vocabulary for action tokens (built if None)
        prop_embedder:    PropertyEmbedder (built if None)
        max_objects:      pad / truncate object list to this length
        max_lang_tokens:  pad / truncate language tokens
        max_action_len:   pad / truncate action sequence
        num_joints:       expected skeleton joint count
    """

    def __init__(
        self,
        json_path: str,
        lang_vocab: Optional[Vocabulary] = None,
        action_vocab: Optional[Vocabulary] = None,
        prop_embedder: Optional[PropertyEmbedder] = None,
        max_objects: int = 16,
        max_lang_tokens: int = 32,
        max_action_len: int = 16,
        num_joints: int = 18,
    ):
        super().__init__()
        with open(json_path) as f:
            self.raw_data: List[Dict[str, Any]] = json.load(f)

        self.max_objects = max_objects
        self.max_lang_tokens = max_lang_tokens
        self.max_action_len = max_action_len
        self.num_joints = num_joints

        # Build vocabularies from data if not provided
        if prop_embedder is None:
            prop_embedder = PropertyEmbedder()
            prop_embedder.fit([s["objects"] for s in self.raw_data])
        self.prop_embedder = prop_embedder

        if lang_vocab is None:
            lang_tokens_set: set = set()
            for s in self.raw_data:
                lang_tokens_set.update(s["instruction"].lower().split())
            lang_vocab = Vocabulary(sorted(lang_tokens_set))
        self.lang_vocab = lang_vocab

        if action_vocab is None:
            action_tokens_set: set = set()
            for s in self.raw_data:
                action_tokens_set.update(s["action_sequence"])
            action_vocab = Vocabulary(sorted(action_tokens_set))
        self.action_vocab = action_vocab

    def __len__(self) -> int:
        return len(self.raw_data)

    def _pad_single_pose(self, raw: List[List[float]]) -> List[List[float]]:
        """Pad / truncate one pose frame to num_joints."""
        if len(raw) < self.num_joints:
            raw = raw + [[0.0, 0.0, 0.0]] * (self.num_joints - len(raw))
        return raw[: self.num_joints]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_data[idx]

        # --- Pose: (T, K, 3) where T=1 (single gesture) or T=2 (dual gesture) ---
        # Supports both old `pose` field (single frame) and new `poses` field.
        if "poses" in sample:
            frames = [self._pad_single_pose(f) for f in sample["poses"]]
        else:
            frames = [self._pad_single_pose(sample["pose"])]
        pose_t = torch.tensor(frames, dtype=torch.float32)  # (T, K, 3)

        # --- Objects ---
        # Continuous: (N, 6)  — [x, y, z, roll, pitch, yaw]
        # Categorical: (N, 2) — [color_id, shape_id] using *language* vocab IDs
        # so that "green" in the instruction and "green" as a color share the
        # same embedding representation in the model.
        obj_cont = []
        obj_cat  = []
        unk_id   = self.lang_vocab.token2id[Vocabulary.UNK]
        for obj in sample["objects"][: self.max_objects]:
            pos = obj.get("position",    [0.0, 0.0, 0.0])
            ori = obj.get("orientation", [0.0, 0.0, 0.0])
            obj_cont.append(pos + ori)
            color_id = self.lang_vocab.token2id.get(obj.get("color", ""), unk_id)
            shape_id = self.lang_vocab.token2id.get(obj.get("shape", ""), unk_id)
            obj_cat.append([color_id, shape_id])
        n_real = len(obj_cont)
        pad_id = self.lang_vocab.pad_id
        while len(obj_cont) < self.max_objects:
            obj_cont.append([0.0] * 6)
            obj_cat.append([pad_id, pad_id])
        objects_t = torch.tensor(obj_cont, dtype=torch.float32)  # (N, 6)
        obj_cat_t = torch.tensor(obj_cat,  dtype=torch.long)     # (N, 2)
        obj_mask  = torch.zeros(self.max_objects, dtype=torch.bool)
        obj_mask[n_real:] = True

        # --- Language tokens ---
        words = sample["instruction"].lower().split()
        lang_ids = self.lang_vocab.encode(words)
        n_lang = min(len(lang_ids), self.max_lang_tokens)
        lang_padded = lang_ids[:n_lang] + [self.lang_vocab.pad_id] * (
            self.max_lang_tokens - n_lang
        )
        lang_t = torch.tensor(lang_padded, dtype=torch.long)
        lang_mask = torch.zeros(self.max_lang_tokens, dtype=torch.bool)
        lang_mask[n_lang:] = True

        # --- Target object index ---
        target_t = torch.tensor(sample["target_object_idx"], dtype=torch.long)

        # --- Reference object index (-1 means single-gesture episode) ---
        ref_idx = sample.get("reference_object_idx", -1)
        ref_t   = torch.tensor(ref_idx, dtype=torch.long)

        # --- Action sequence (with START / END) ---
        raw_actions = sample["action_sequence"]
        action_ids = (
            [self.action_vocab.start_id]
            + self.action_vocab.encode(raw_actions)
            + [self.action_vocab.end_id]
        )
        n_act = min(len(action_ids), self.max_action_len)
        action_padded = action_ids[:n_act] + [self.action_vocab.pad_id] * (
            self.max_action_len - n_act
        )
        action_t    = torch.tensor(action_padded, dtype=torch.long)
        action_mask = torch.zeros(self.max_action_len, dtype=torch.bool)
        action_mask[n_act:] = True

        return {
            "pose":         pose_t,       # (T, K, 3) — T=1 or T=2
            "objects":      objects_t,    # (max_objects, 6)  continuous
            "obj_cat":      obj_cat_t,    # (max_objects, 2)  int [color_id, shape_id]
            "obj_mask":     obj_mask,     # (max_objects,)
            "lang_tokens":  lang_t,       # (max_lang_tokens,)
            "lang_mask":    lang_mask,    # (max_lang_tokens,)
            "target_idx":   target_t,     # scalar
            "ref_idx":      ref_t,        # scalar (-1 = single gesture)
            "action_input": action_t,     # (max_action_len,)
            "action_mask":  action_mask,  # (max_action_len,)
        }

    def get_action_target(self, action_input: torch.Tensor) -> torch.Tensor:
        """Shift action_input left by 1 to create teacher-forcing targets."""
        target = action_input.clone()
        target[..., :-1] = action_input[..., 1:]
        target[..., -1] = self.action_vocab.pad_id
        return target


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate that pads the pose temporal dimension so batches with
    mixed single-gesture (T=1) and dual-gesture (T=2) samples can be
    stacked into a single (B, T_max, K, 3) tensor.
    """
    T_max = max(item["pose"].shape[0] for item in batch)
    K     = batch[0]["pose"].shape[1]

    padded_poses = []
    for item in batch:
        T = item["pose"].shape[0]
        if T < T_max:
            pad = torch.zeros(T_max - T, K, 3)
            padded_poses.append(torch.cat([item["pose"], pad], dim=0))
        else:
            padded_poses.append(item["pose"])

    result = {k: torch.stack([item[k] for item in batch]) for k in batch[0] if k != "pose"}
    result["pose"] = torch.stack(padded_poses)   # (B, T_max, K, 3)
    return result
