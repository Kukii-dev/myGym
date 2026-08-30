"""
Dataset generator for the Reasoner module.

For each episode this generator:
  1. Randomly picks an object shape (cube/sphere/cylinder/prism/pyramid)
  2. Resets the env  (spawns 3 objects of that shape)
  3. Re-assigns DISTINCT random colors to every object
     (overriding the env's highlight-coloring which makes all targets red)
  4. Randomly picks an action_type and re-generates the NL instruction
  5. Maps action_type → action sequence token list
  6. Records pose, objects, instruction, target_idx, action_sequence

Usage:
    python -m myGym.reasoner.generate_dataset \\
        --config  myGym/configs/train_gesture_tiago.json \\
        --episodes 500 \\
        --output  myGym/reasoner/dataset.json

Flags:
    --config      Path to a GymEnv JSON config with task_type="reach_gesture"
    --episodes    Number of episodes to collect   (default 500)
    --output      Output JSON file path
    --sim_steps   Extra simulation steps after reset for IK to settle (default 30)
    --gui         1 = show PyBullet GUI
    --seed        Random seed (default 42)
    --test        Run 3 episodes with full debug output then exit
"""

import argparse
import copy
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import commentjson as _json_mod
except ImportError:
    import json as _json_mod

import myGym.utils.colors as cs
from myGym.envs.natural_language import GestureLanguageGenerator


# ---------------------------------------------------------------------------
# Variability tables
# ---------------------------------------------------------------------------

# Geometric shapes available in myGym/envs/objects/geometric/urdf/
AVAILABLE_SHAPES = ["cube", "sphere", "cylinder", "prism", "pyramid"]

# Colors that can be uniquely read back via cs.rgba_to_name()
AVAILABLE_COLORS = [k for k in cs.OPAQUE_COLOR_DICT.keys()]

# Single-gesture: action_type → robot action sequence
SINGLE_GESTURE_ACTION_MAP: Dict[str, List[str]] = {
    "point":   ["A"],
    "pick_up": ["A", "G"],
    "move":    ["A", "G", "M", "D"],
}

# Two-gesture: action_type → robot action sequence
TWO_GESTURE_ACTION_MAP: Dict[str, List[str]] = {
    "move_to": ["A", "G", "M", "D"],
}


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _get_human_pose(ue) -> List[List[float]]:
    """World-space [x,y,z] for every human model link. Index 0 = base."""
    p = ue.p
    bid = ue.human.body_id
    base_pos, _ = p.getBasePositionAndOrientation(bid)
    joints = [list(base_pos)]
    for i in range(ue.human.n_joints):
        joints.append(list(p.getLinkState(bid, i)[0]))
    return joints


def _classify_size(dims: List[float]) -> str:
    vol = dims[0] * dims[1] * dims[2]
    if vol < 5e-4:
        return "small"
    if vol < 5e-3:
        return "medium"
    return "large"


def _object_to_dict(obj, color_name: str) -> Dict[str, Any]:
    """Serialise an EnvObject using the SUPPLIED color_name (not the stored one)."""
    pos   = list(obj.get_position())
    euler = list(obj.get_orientation_euler())
    shape = obj.get_name().split("_")[0].lower()
    try:
        size = _classify_size(obj.get_cuboid_dimensions())
    except Exception:
        size = "unknown"
    return {
        "position":    pos,
        "orientation": euler,
        "color":       color_name,
        "shape":       shape,
        "size":        size,
        "texture":     "smooth",
    }


# ---------------------------------------------------------------------------
# Per-episode collection
# ---------------------------------------------------------------------------

def _find_object_index(target_obj, all_objs: list) -> int:
    try:
        return all_objs.index(target_obj)
    except ValueError:
        uid = target_obj.uid
        return next((i for i, o in enumerate(all_objs) if o.uid == uid), 0)


def _settle(ue, sim_steps: int):
    for _ in range(sim_steps):
        ue.p.stepSimulation()


def _collect_episode(
    env,
    rng: random.Random,
    sim_steps: int,
    verbose: bool,
    dual_gesture_prob: float = 0.5,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Reset the env and collect one dataset sample.

    With probability `dual_gesture_prob` the human performs TWO gestures:
      gesture 1 → points at the target object   (pose frame 0)
      gesture 2 → points at a reference object  (pose frame 1)
    Otherwise a single-gesture episode is collected (pose frame 0 only).

    The dataset stores `poses: [[frame0_joints], ...]` (list of T frames).
    """
    ue = env.unwrapped

    # 1. Pick random shape, update env config and phrase generator
    shape = rng.choice(AVAILABLE_SHAPES)
    for obj_def in ue.task_objects_dict.get("goal", []):
        obj_def["obj_name"] = shape
    ue.gesture_nl.OBJECT_TYPE = shape

    # 2. Reset — spawns objects, human points at one randomly
    env.reset()

    if not getattr(ue, "gesture_cubes", None):
        return None, "gesture_cubes empty after reset"
    if getattr(ue, "human_target_object", None) is None:
        return None, "human_target_object is None"

    # 3. Re-assign distinct random colors (overrides env highlight coloring)
    all_objs = ue.gesture_cubes
    n = len(all_objs)
    chosen_colors = rng.sample(AVAILABLE_COLORS, min(n, len(AVAILABLE_COLORS)))
    if len(chosen_colors) < n:
        chosen_colors = (chosen_colors * ((n // len(chosen_colors)) + 1))[:n]
    color_map = {id(o): c for o, c in zip(all_objs, chosen_colors)}
    for obj, color_name in zip(all_objs, chosen_colors):
        obj.set_color(cs.name_to_rgba(color_name))

    target_obj = ue.human_target_object
    target_idx = _find_object_index(target_obj, all_objs)

    # 4. Let IK settle after the first (target) gesture, then snapshot pose 1
    _settle(ue, sim_steps)
    pose_frame1 = _get_human_pose(ue)

    # 5. Decide: single or dual gesture?
    dual = rng.random() < dual_gesture_prob
    ref_obj   = None
    ref_idx   = -1
    pose_frame2 = None

    if dual and n >= 2:
        # Pick a reference object that is NOT the target
        candidates = [o for o in all_objs if id(o) != id(target_obj)]
        ref_obj = rng.choice(candidates)
        ref_idx = _find_object_index(ref_obj, all_objs)

        # Human performs second gesture: arm moves to point at reference object
        ue.human.point_finger_at(
            position=ref_obj.get_position(),
            use_pointing_ik=True,
        )
        _settle(ue, sim_steps)
        pose_frame2 = _get_human_pose(ue)

    # 6. Build object records
    objects_data = [_object_to_dict(o, color_map[id(o)]) for o in all_objs]

    # 7. Generate NL instruction
    if dual and ref_obj is not None:
        action_type = rng.choice(list(GestureLanguageGenerator.TWO_GESTURE_TEMPLATES.keys()))
        phrase_meta = ue.gesture_nl.generate_two_gesture_phrase(
            target_obj=target_obj,
            ref_obj=ref_obj,
            action_type=action_type,
        )
        action_sequence = TWO_GESTURE_ACTION_MAP.get(action_type, ["A", "G", "M", "D"])
    else:
        action_type = rng.choice(list(GestureLanguageGenerator.PHRASE_TEMPLATES.keys()))
        phrase_meta = ue.gesture_nl.generate_phrase(
            target_obj=target_obj,
            all_objects=all_objs,
            action_type=action_type,
        )
        action_sequence = SINGLE_GESTURE_ACTION_MAP.get(action_type, ["A", "G"])

    instruction = phrase_meta["phrase"]

    # 8. Assemble sample — `poses` is a list of T frames (T=1 or T=2)
    poses = [pose_frame1] if pose_frame2 is None else [pose_frame1, pose_frame2]

    sample = {
        "poses":               poses,
        "objects":             objects_data,
        "instruction":         instruction,
        "target_object_idx":   target_idx,
        "reference_object_idx": ref_idx,
        "action_sequence":     action_sequence,
        "num_gestures":        len(poses),
    }

    if verbose:
        obj_desc = [f"{o['color']} {o['shape']}" for o in objects_data]
        print(f"    shape        : {shape}")
        print(f"    objects      : {obj_desc}")
        print(f"    target       : [{target_idx}] {objects_data[target_idx]['color']}")
        if ref_idx >= 0:
            print(f"    reference    : [{ref_idx}] {objects_data[ref_idx]['color']}")
        print(f"    gestures     : {len(poses)}")
        print(f"    action_type  : {action_type}")
        print(f"    instruction  : \"{instruction}\"")
        print(f"    action_seq   : {action_sequence}")
        print(f"    joints/frame : {len(pose_frame1)}")

    return sample, None


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _build_env(config_path: str, gui: bool, seed: int):
    import gymnasium as gym
    import myGym.envs  # registers Gym-v0, CrowWorkspaceEnv-v0, etc.

    with open(config_path) as f:
        cfg = _json_mod.load(f)

    cfg["gui"]       = 1 if gui else 0
    cfg["visualize"] = 0
    cfg["visgym"]    = 0
    cfg["seed"]      = seed
    cfg["render"]    = "opengl" if gui else "offscreen"

    if cfg.get("task_type") not in ("reach_gesture", "AG"):
        print(f"[WARNING] task_type='{cfg.get('task_type')}' is not 'reach_gesture'.")

    env = gym.make(
        cfg["env_name"],
        workspace              = cfg["workspace"],
        robot                  = cfg["robot"],
        robot_action           = cfg.get("robot_action", "joints"),
        robot_init_joint_poses = cfg.get("robot_init", "default"),
        max_velocity           = cfg.get("max_velocity", 3),
        max_force              = cfg.get("max_force", 200),
        task_type              = cfg["task_type"],
        task_objects           = cfg["task_objects"],
        observation            = cfg.get("observation", {
                                     "actual_state": "obj_6D",
                                     "goal_state":   "obj_6D",
                                 }),
        distractors            = cfg.get("distractors", {
                                     "list": None, "moveable": 0,
                                     "constant_speed": 0, "movement_dims": 3,
                                     "movement_endpoints": [-0.3, 0.3, 0.4, 0.7, 0.1, 0.3],
                                 }),
        used_objects           = cfg.get("used_objects", {"num_range": [0, 0], "obj_list": []}),
        active_cameras         = cfg.get("camera", 0),
        color_dict             = cfg.get("color_dict", {}),
        distance_type          = cfg.get("distance_type", "euclidean"),
        vae_path               = cfg.get("vae_path"),
        yolact_path            = cfg.get("yolact_path"),
        yolact_config          = cfg.get("yolact_config"),
        action_repeat          = cfg.get("action_repeat", 1),
        render_on              = gui,
        visualize              = cfg["visualize"],
        visgym                 = cfg["visgym"],
        gui_on                 = gui,
    )

    ue = env.unwrapped
    if not getattr(ue, "reach_gesture", False):
        print("[WARNING] env.reach_gesture is False — human/gesture system inactive.")
    if not hasattr(ue, "human"):
        print("[ERROR] env.human not found. Wrong workspace or task_type?")

    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--episodes",  type=int, default=500)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--sim_steps", type=int, default=30)
    parser.add_argument("--gui",       type=int, default=0)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--test",      action="store_true",
                        help="Run 3 episodes with full debug output then exit")
    args = parser.parse_args()

    if args.test:
        args.episodes = 3

    print(f"Config    : {args.config}")
    print(f"Episodes  : {args.episodes}")
    print(f"Output    : {args.output}")
    print(f"Shapes    : {AVAILABLE_SHAPES}")
    print(f"Colors    : {AVAILABLE_COLORS}")
    print(f"Act types (1-gesture): {list(SINGLE_GESTURE_ACTION_MAP.keys())}")
    print(f"Act types (2-gesture): {list(TWO_GESTURE_ACTION_MAP.keys())}")
    print()

    print("Building environment...")
    env = _build_env(args.config, gui=bool(args.gui), seed=args.seed)
    print(f"Environment ready: {env.unwrapped.__class__.__name__}\n")

    rng = random.Random(args.seed)
    dataset: List[Dict] = []
    errors:  List[str]  = []

    for ep in range(args.episodes):
        verbose = args.test or ep < 3
        print(f"Episode {ep + 1:4d}/{args.episodes} ...", end="\n" if verbose else " ", flush=True)

        try:
            sample, reason = _collect_episode(env, rng, args.sim_steps, verbose)

            if sample is None:
                msg = f"SKIP — {reason}"
                print(msg)
                errors.append(f"ep{ep}: {msg}")
                continue

            dataset.append(sample)
            tgt = sample["objects"][sample["target_object_idx"]]
            if not verbose:
                print(f"OK  [{tgt['color']:8s} {tgt['shape']:10s}] "
                      f"act={sample['action_sequence']}  "
                      f"\"{sample['instruction']}\"")

        except Exception as exc:
            tb = traceback.format_exc()
            msg = f"ERROR — {exc or type(exc).__name__}"
            print(msg)
            errors.append(f"ep{ep}: {msg}")
            if verbose:
                print("  --- traceback ---")
                for line in tb.splitlines():
                    print("  " + line)
            continue

        if args.test and ep == 2:
            break

    env.close()

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Collected : {len(dataset)} samples")
    print(f"Failed    : {len(errors)}")

    if not dataset:
        print("\n[ERROR] No samples collected.")
        if errors:
            print("First error:", errors[0])
        return

    # Dataset statistics
    from collections import Counter
    shapes   = Counter(o["shape"] for s in dataset for o in s["objects"])
    colors   = Counter(o["color"] for s in dataset for o in s["objects"])
    act_seqs = Counter(tuple(s["action_sequence"]) for s in dataset)
    print(f"\nDataset statistics:")
    print(f"  Shapes      : {dict(shapes)}")
    print(f"  Colors      : {dict(colors)}")
    print(f"  Action seqs : {dict(act_seqs)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved to : {args.output}")


if __name__ == "__main__":
    main()
