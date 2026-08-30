import copy
import itertools
import re
from enum import Enum, auto
from typing import Tuple, List, Dict, Optional

import numpy as np

from myGym.envs.env_object import EnvObject
import myGym.utils.colors as cs


def _filter_out_none(iterable):
    return [e for e in iterable if e is not None]


def _unpack_1_or_2_element_tuple(t):
    return t if len(t) == 2 else (t[0], None)


def _remove_first_word(s):
    return s[s.find(" ") + 1:]


def _remove_last_word(s):
    return s[:s.rfind(" ")]


class TaskType(Enum):
    REACH = auto(),
    PUSH = auto(),
    PNP = auto(),
    PNPROT = auto(),
    PNPSWIPE = auto(),
    PNPBGRIP = auto(),
    THROW = auto(),
    POKE = auto(),
    PRESS = auto(),
    TURN = auto(),
    SWITCH = auto(),
    FMRT = auto(),
    FMOT = auto(),
    COMPOSITIONAL = auto(),

    def to_string(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_string(task: str):
        for entry in TaskType:
            if entry.name.lower() == task:
                return entry
        msg = f"Unknown task type: {task}"
        raise Exception(msg)

    @staticmethod
    def get_pattern_reach_task_types() -> List:
        return [TaskType.REACH]

    @staticmethod
    def get_pattern_press_task_types() -> List:
        return [TaskType.PRESS, TaskType.TURN, TaskType.SWITCH]

    @staticmethod
    def get_pattern_push_task_types() -> List:
        return [TaskType.PUSH, TaskType.PNP, TaskType.PNPROT, TaskType.PNPSWIPE, TaskType.PNPBGRIP, TaskType.THROW, TaskType.POKE, TaskType.FMRT, TaskType.FMOT]


# ---------------------------------------------------------------------------
# GestureLanguageGenerator — natural language for gesture-based training
# ---------------------------------------------------------------------------

class GestureLanguageGenerator:
    """
    Generates natural language commands for gesture-based TIAgo training.

    On every episode reset the environment calls ``generate_phrase()`` which:
      1. Picks a random action type (pick_up, move, reach, push, point).
      2. Fills in a randomly chosen template with the actual object colors,
         spatial relations, and randomised articles / synonyms.
      3. Returns a metadata dict that fully describes the sentence so that
         downstream NLP models can be trained on it.

    Parameters:
        :param seed: (int) Seed for the random generator
    """

    # -- phrase templates -------------------------------------------------------
    # Placeholders:
    #   {color}      – target object color        {obj}       – object type (cube)
    #   {ref_color}  – reference object color      {direction} – left / right
    #   {dir_of}     – "to the left of" / …       {a} / {a2}  – random article
    PHRASE_TEMPLATES: Dict[str, List[str]] = {
        "pick_up": [
            "pick up this {color} {obj}",
            "pick up the {color} {obj}",
            "pick up the {obj} next to the {ref_color} {obj}",
            "pick up the {color} {obj} {dir_of} the {ref_color} {obj}",
        ],
        "move": [
            "move this {color} {obj} to the {direction}",
            "move the {color} {obj} next to the {ref_color} {obj}",
            "put this {color} {obj} next to the {ref_color} {obj}",
            "put the {color} {obj} {dir_of} the {ref_color} {obj}",
        ],
        "point": [
            "point at this {color} {obj}",
            "point at the {color} {obj}",
            "point at the {obj} next to the {ref_color} {obj}",
            "point at the {obj} {dir_of} the {ref_color} {obj}",
        ],
    }

    DIRECTIONS = ["left", "right"]
    DIRECTION_PREPOSITIONS = ["to the left of", "to the right of"]
    ARTICLES = ["the ", "this ", "that "]
    # Reference objects in single-gesture phrases get a definite article only —
    # "this"/"that" would imply a second gesture the human doesn't make.
    REF_ARTICLES = ["the "]
    OBJECT_TYPE = "cube"

    # Two-gesture templates: BOTH {color} and {ref_color} objects use deictic
    # articles because the human physically performs two separate pointing gestures.
    TWO_GESTURE_TEMPLATES: Dict[str, List[str]] = {
        "move_to": [
            "first pick up this {color} {obj} then move it next to this {ref_color} {obj}",
            "first pick up this {color} {obj} then put it next to this {ref_color} {obj}",
            "first point at this {color} {obj} then move it to this {ref_color} {obj}",
            "first point at this {color} {obj} then put it next to this {ref_color} {obj}",
            "put this {color} {obj} next to this {ref_color} {obj}",
            "move this {color} {obj} to this {ref_color} {obj}",
            "place this {color} {obj} next to this {ref_color} {obj}",
        ],
    }

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _get_object_color(obj: EnvObject) -> str:
        color_name = cs.rgba_to_name(obj.get_color_rgba())
        return color_name if color_name else "unknown"

    @staticmethod
    def _get_spatial_relation(obj: EnvObject, ref_obj: EnvObject) -> str:
        pos = np.array(obj.get_position())
        ref_pos = np.array(ref_obj.get_position())
        return "left" if pos[0] > ref_pos[0] else "right"

    # -- public API -------------------------------------------------------------

    def generate_phrase(
        self,
        target_obj: EnvObject,
        all_objects: List[EnvObject],
        action_type: Optional[str] = None,
    ) -> Dict:
        """
        Generate a random natural language phrase and structured metadata.

        Parameters:
            :param target_obj: (EnvObject) The object the action refers to
            :param all_objects: (List[EnvObject]) All objects currently in the scene
            :param action_type: (str, optional) Force action type; random if None

        Returns:
            :return: (dict) with keys
                - phrase            (str)  generated sentence
                - action_type       (str)  pick_up | move | reach | push | point
                - target_color      (str)  color name of target object
                - target_position   (list) [x, y, z]
                - ref_color         (str|None)  color of reference object
                - ref_position      (list|None) [x, y, z] of reference object
                - direction         (str|None)  left / right
                - tokens            (list) whitespace-split tokens
                - object_type       (str)  e.g. "cube"
                - num_tokens        (int)  len(tokens)
        """
        if action_type is None:
            action_type = str(self.rng.choice(list(self.PHRASE_TEMPLATES.keys())))

        templates = self.PHRASE_TEMPLATES[action_type]
        target_color = self._get_object_color(target_obj)
        target_pos = list(target_obj.get_position())

        # reference object (any object that is not the target)
        other_objects = [o for o in all_objects if o is not target_obj]
        ref_obj = self.rng.choice(other_objects) if other_objects else None
        ref_color = self._get_object_color(ref_obj) if ref_obj else None
        ref_pos = list(ref_obj.get_position()) if ref_obj else None

        # spatial direction
        if ref_obj is not None:
            direction = self._get_spatial_relation(target_obj, ref_obj)
        else:
            direction = str(self.rng.choice(self.DIRECTIONS))
        dir_of = self.DIRECTION_PREPOSITIONS[self.DIRECTIONS.index(direction)]

        # filter templates that need a reference when none is available
        usable = [t for t in templates
                  if ref_obj is not None or
                  ("{ref_color}" not in t and "{dir_of}" not in t and "{a2}" not in t)]
        if not usable:
            usable = [f"{action_type.replace('_', ' ')} {{a}}{{color}} {{obj}}"]

        template = str(self.rng.choice(usable))
        a  = str(self.rng.choice(self.ARTICLES))      # target — deictic ok
        a2 = str(self.rng.choice(self.REF_ARTICLES))  # reference — always "the"

        phrase = template.format(
            color=target_color,
            obj=self.OBJECT_TYPE,
            ref_color=ref_color or "",
            direction=direction,
            dir_of=dir_of,
            a=a,
            a2=a2,
        )
        phrase = re.sub(r" +", " ", phrase).strip()
        tokens = phrase.split()

        return {
            "phrase": phrase,
            "action_type": action_type,
            "target_color": target_color,
            "target_position": target_pos,
            "ref_color": ref_color,
            "ref_position": ref_pos,
            "direction": direction,
            "tokens": tokens,
            "object_type": self.OBJECT_TYPE,
            "num_tokens": len(tokens),
        }

    def generate_batch(
        self,
        target_obj: EnvObject,
        all_objects: List[EnvObject],
        n: int = 5,
    ) -> List[Dict]:
        """
        Generate *n* phrase variants for the same scene, cycling through action types.
        """
        action_types = list(self.PHRASE_TEMPLATES.keys())
        return [
            self.generate_phrase(target_obj, all_objects,
                                 action_type=action_types[i % len(action_types)])
            for i in range(n)
        ]

    def generate_two_gesture_phrase(
        self,
        target_obj: EnvObject,
        ref_obj: EnvObject,
        action_type: Optional[str] = None,
    ) -> Dict:
        """
        Generate a phrase for a two-gesture scenario where the human physically
        points at *both* target_obj (gesture 1) and ref_obj (gesture 2).

        Both objects are referred to with deictic "this" because the human
        actually performs two distinct pointing gestures.

        Parameters:
            :param target_obj:  (EnvObject) Object the first gesture points at
            :param ref_obj:     (EnvObject) Object the second gesture points at
            :param action_type: (str, optional) "move_to" | "push_to"; random if None

        Returns:
            :return: (dict) with keys phrase, action_type, target_color,
                     ref_color, target_position, ref_position, tokens, num_tokens
        """
        if action_type is None:
            action_type = str(self.rng.choice(list(self.TWO_GESTURE_TEMPLATES.keys())))

        templates = self.TWO_GESTURE_TEMPLATES[action_type]
        template  = str(self.rng.choice(templates))

        target_color = self._get_object_color(target_obj)
        ref_color    = self._get_object_color(ref_obj)

        phrase = template.format(
            color=target_color,
            ref_color=ref_color,
            obj=self.OBJECT_TYPE,
        )
        phrase = re.sub(r" +", " ", phrase).strip()
        tokens = phrase.split()

        return {
            "phrase":           phrase,
            "action_type":      action_type,
            "target_color":     target_color,
            "ref_color":        ref_color,
            "target_position":  list(target_obj.get_position()),
            "ref_position":     list(ref_obj.get_position()),
            "tokens":           tokens,
            "num_tokens":       len(tokens),
            "num_gestures":     2,
        }


# ---------------------------------------------------------------------------
# Legacy helpers & classes (kept for backward-compatibility with nl_mode)
# ---------------------------------------------------------------------------

class VirtualObject:
    def __init__(self, obj: EnvObject):
        self.obj: EnvObject = obj
        name = self.obj.get_name()
        self.name = name if "_" not in name else name.split("_", 1)[0]
        self.properties = " ".join(_filter_out_none([cs.rgba_to_name(obj.get_color_rgba())]))

    def __deepcopy__(self, memo={}):
        cp = VirtualObject(self.obj)
        cp.__dict__.update(self.__dict__)
        return cp

    def get_env_object(self) -> EnvObject:
        return self.obj

    def get_name(self) -> str:
        return "the " + self.name

    def get_properties(self) -> str:
        return "the " + self.properties

    def get_name_with_properties(self) -> str:
        return "the " + self.properties + " " + self.name

    @staticmethod
    def extract_object_from_name_with_properties(desc: str, objects: List):
        color_name = _remove_last_word(_remove_first_word(desc))
        object_matches = [o for o in objects if o.properties == color_name]
        if len(object_matches) != 1:
            msg = f"Cannot uniquely determine object, there are {len(object_matches)} objects with description \"{desc}\""
            raise Exception(msg)
        return object_matches[0]

    def get_name_as_unknown_object_with_properties(self) -> str:
        return "the " + self.properties + " object"

    @staticmethod
    def extract_objects_from_unknown_object_with_properties(desc: str, objects: List):
        color_name = _remove_last_word(_remove_first_word(desc))
        object_matches = [o for o in objects if o.properties == color_name]
        if len(object_matches) == 0:
            print(" ".join([o.properties for o in objects]))
            msg = f"There are no objects with description \"{desc}\""
            raise Exception(msg)
        return object_matches

    def get_position(self) -> np.array:
        return np.array(self.obj.get_position())


class VirtualEnv:
    """
    Internal class for WIP multistep task generation (to virtually simulate object movement).
    """
    def __init__(self, env):
        self.env = env
        self.task_type: TaskType = TaskType.from_string(env.task_type)
        self.objects: List[VirtualObject] = []
        self.real_object_indices: List[int] = []
        self.dummy_object_indices: List[int] = []
        self.set_objects(task_objects=env.task_objects)

    def set_objects(self, task_objects=None, init_goal_objects=None, all_objects=None):
        if bool(task_objects) + bool(init_goal_objects) + bool(all_objects) > 1:
            raise Exception("The only one argument must be passed")

        if task_objects:
            self.objects: List[VirtualObject] = [VirtualObject(o) if isinstance(o, EnvObject) else None for o in self.env.get_task_objects(with_none=True)]
            self.real_object_indices = list(
                range(1, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_reach_task_types()
                else (
                    range(0, len(self.objects), 2) if self.get_task_type() in TaskType.get_pattern_push_task_types()
                    else []
                )
            )
        elif init_goal_objects:
            init, goal = init_goal_objects
            self.objects = list(map(VirtualObject, init + goal))
            if TaskType.from_string(self.env.task_type) in TaskType.get_pattern_push_task_types():
                if len(init) == 0 or len(goal) == 0:
                    raise Exception("Not enough real or dummy objects (every group must have at least 1 object)!")
                self.real_object_indices = list(range(len(init)))
                self.dummy_object_indices = list(range(len(init), len(init) + len(goal)))
            else:
                if len(goal) == 0:
                    raise Exception("Not enough real objects!")
                self.real_object_indices = list(range(len(goal)))
        elif all_objects:
            self.objects = list(map(VirtualObject, all_objects))

    def __copy__(self):
        cp = VirtualEnv(self.env)
        cp.__dict__.update(self.__dict__)
        cp.objects = copy.deepcopy(self.objects)
        return cp

    def get_env(self):
        return self.env

    def get_task_type(self) -> TaskType:
        return self.task_type

    def _get_objects(self, indices) -> List[VirtualObject]:
        return [self.objects[i] for i in indices]

    def get_real_objects(self) -> List[VirtualObject]:
        return self._get_objects(self.real_object_indices)

    def get_dummy_objects(self) -> List[VirtualObject]:
        return self._get_objects(self.dummy_object_indices)

    def get_all_objects(self, excluding=None) -> List[VirtualObject]:
        return self.objects if not excluding else [o for o in self.objects if o not in excluding]

    def _get_all_objects_in_relation(self, obj: VirtualObject, relation: str) -> List[VirtualObject]:
        objects = []
        p1 = obj.get_position()

        for o in self.get_all_objects(excluding=[obj]):
            if o is not obj:
                p2 = o.get_position()
                if relation == "left" and p1[0] > p2[0] or relation == "right" and p1[0] < p2[0]:
                    objects.append(o)

        return objects

    def get_all_objects_left_of(self, obj: VirtualObject) -> List[VirtualObject]:
        return self._get_all_objects_in_relation(obj, "left")

    def get_all_objects_right_of(self, obj: VirtualObject) -> List[VirtualObject]:
        return self._get_all_objects_in_relation(obj, "right")

    def get_subtask_objects(self) -> List[Tuple]:
        return [tuple(_filter_out_none([self.objects[i], self.objects[i + 1]]))
                for i in range(0, len(self.objects), 2)]

    def get_current_subtask_idx(self) -> int:
        return self.env.task.current_task


class NaturalLanguage:
    """
    Legacy NL class kept for backward compatibility with nl_mode.
    For gesture training, use GestureLanguageGenerator instead.
    """
    def __init__(self, env, seed=0):
        self.venv: VirtualEnv = VirtualEnv(env)
        self.current_subtask_description: str or None = None
        self.rng = np.random.default_rng(seed)

    def get_venv(self) -> VirtualEnv:
        return self.venv

    def set_current_subtask_description(self, desc: str):
        self.current_subtask_description = desc

    def get_previously_generated_subtask_description(self) -> str:
        return self.current_subtask_description

    @staticmethod
    def _form_subtask_description(venv: VirtualEnv, *objects_descriptions, task_type: TaskType = None) -> str:
        task_type = venv.get_task_type() if task_type is None else task_type
        d1, d2 = _unpack_1_or_2_element_tuple(objects_descriptions)

        if task_type is TaskType.REACH:
            tokens = ["reach", d1]
        elif task_type is TaskType.PRESS:
            tokens = ["press", d1]
        elif task_type is TaskType.TURN:
            tokens = ["turn", d1]
        elif task_type is TaskType.SWITCH:
            tokens = ["switch", d1]
        elif task_type is TaskType.PUSH:
            tokens = ["push", d1, "to the same position as", d2]
        elif task_type is TaskType.PNP:
            tokens = ["pick", d1, "and place it to the same position as", d2]
        elif task_type is TaskType.PNPROT:
            tokens = ["pick", d1, "and rotate it to the same position as", d2]
        elif task_type is TaskType.PNPSWIPE:
            tokens = ["swipe", d1, "along the line to the position of", d2]
        elif task_type is TaskType.PNPBGRIP:
            bgrip = " with mechanic gripper" if "bgrip" in venv.get_env().robot.get_name() else ""
            tokens = ["pick", d1, "and place it" + bgrip, d2]
        elif task_type is TaskType.THROW:
            tokens = ["throw", d1, "to the same position as", d2]
        elif task_type is TaskType.POKE:
            tokens = ["poke", d1, "to the same position as", d2]
        else:
            exc = f"Unknown task type {task_type}"
            raise Exception(exc)

        return " ".join(tokens)

    @staticmethod
    def _decompose_subtask_description(desc: str):
        if desc.startswith("reach"):
            return TaskType.REACH, _remove_first_word(desc)
        elif desc.startswith("push") or desc.startswith("throw") or desc.startswith("poke"):
            task_type = TaskType.PUSH if desc.startswith("push") else (TaskType.THROW if desc.startswith("throw") else TaskType.POKE)
            return task_type, _remove_first_word(desc).split(" to the same position as ")
        if desc.startswith("pick") and "rotate" not in desc:
            return TaskType.PNP, _remove_first_word(desc).split(" and place it to the same position as ")
        elif desc.startswith("pick") and "rotate" in desc:
            return TaskType.PNPROT, _remove_first_word(desc).split(" and rotate it to the same position as ")
        elif desc.startswith("swipe"):
            return TaskType.PNPSWIPE, _remove_first_word(desc).split(" along the line to the position of ")
        else:
            msg = f"Cannot determine the task type: {desc}"
            raise Exception(msg)

    @staticmethod
    def _get_object_descriptions(venv: VirtualEnv, obj: VirtualObject):
        if venv.get_env().reach_gesture:
            return ["here", "there"]

        descs = [obj.get_name_with_properties()]
        for o in venv.get_all_objects_left_of(obj):
            descs.append(" ".join([obj.get_name_as_unknown_object_with_properties(), "right to", o.get_name_with_properties()]))
        for o in venv.get_all_objects_right_of(obj):
            descs.append(" ".join([obj.get_name_as_unknown_object_with_properties(), "left to", o.get_name_with_properties()]))
        return descs

    @staticmethod
    def _extract_object_from_object_description(venv: VirtualEnv, desc: str) -> VirtualObject:
        all_objects = venv.get_all_objects()

        if "left" in desc or "right" in desc:
            is_left = "left" in desc
            descs = desc.split(" left to " if is_left else " right to ")
            d1, d2 = descs[0], descs[1]

            objects_with_same_color = VirtualObject.extract_objects_from_unknown_object_with_properties(d1, all_objects)
            o2 = VirtualObject.extract_object_from_name_with_properties(d2, all_objects)

            if len(objects_with_same_color) == 1:
                return objects_with_same_color[0]
            else:
                objects_in_relation = venv.get_all_objects_left_of(o2) if is_left else venv.get_all_objects_right_of(o2)
                object_matches = list(set(objects_with_same_color) & set(objects_in_relation))

                if len(object_matches) == 1:
                    return object_matches[0]
                else:
                    msg = f"Error, there are {len(object_matches)} objects with description \"{desc}\""
                    raise Exception(msg)
        elif "here" in desc or "there" in desc:
            objects = [vo.get_env_object() for vo in venv.get_all_objects()]
            return VirtualObject(venv.get_env().human.find_object_human_is_pointing_at(objects=objects))
        else:
            return VirtualObject.extract_object_from_name_with_properties(desc, all_objects)

    def generate_subtask_with_random_description(self) -> None:
        task_type = self.venv.get_task_type()
        assert task_type in TaskType.get_pattern_push_task_types() or task_type == TaskType.REACH

        if task_type in TaskType.get_pattern_push_task_types():
            d1 = self.rng.choice(self._get_object_descriptions(self.venv, self.rng.choice(self.venv.get_real_objects())))
            d2 = self.rng.choice(self._get_object_descriptions(self.venv, self.rng.choice(self.venv.get_dummy_objects())))
            self.current_subtask_description = self._form_subtask_description(self.venv, d1, d2)
        else:
            o2 = self.rng.choice(self.venv.get_real_objects())
            env = self.venv.get_env()

            if env.reach_gesture:
                if env.training:
                    for _ in range(1):
                        env.human.point_finger_at(position=o2.get_env_object().get_position())
                        env.p.stepSimulation()
                    o2 = env.human.find_object_human_is_pointing_at(objects=self.venv.get_real_objects())
                else:
                    objects = [vo.get_env_object() for vo in self.venv.get_all_objects()]
                    o2 = VirtualObject(env.choose_goal_object_by_human_with_keys(objects=objects))

            d2 = self.rng.choice(self._get_object_descriptions(self.venv, o2))
            self.current_subtask_description = self._form_subtask_description(self.venv, d2)

    def extract_subtask_info_from_description(self, desc: str) -> Tuple[str, str, int, EnvObject, EnvObject]:
        desc = re.sub(' +', ' ', desc.strip().lower())
        task_type, descs = self._decompose_subtask_description(desc)
        assert task_type in TaskType.get_pattern_push_task_types() or task_type == TaskType.REACH

        if task_type in TaskType.get_pattern_push_task_types():
            d1, d2 = descs[0], descs[1]
            init = self._extract_object_from_object_description(self.venv, d1)
            goal = self._extract_object_from_object_description(self.venv, d2)
        else:
            d2 = descs
            init = None
            goal = self._extract_object_from_object_description(self.venv, d2)

        if task_type is TaskType.REACH or task_type is TaskType.PUSH:
            reward, n_nets = "distance", 1
        else:
            reward, n_nets = task_type.to_string(), 3

        return task_type.to_string(), reward, n_nets, init.get_env_object() if init is not None else init, goal.get_env_object()

    def generate_random_description_for_current_subtask(self) -> None:
        o1, o2 = _unpack_1_or_2_element_tuple(self.venv.get_subtask_objects()[self.venv.get_current_subtask_idx()])
        task_type = self.venv.get_task_type()
        d1 = self.rng.choice(self._get_object_descriptions(self.venv, o1))
        d2 = self.rng.choice(self._get_object_descriptions(self.venv, o2)) if o2 is not None else None
        ds = (d1, d2) if task_type in TaskType.get_pattern_push_task_types() else (d1,)
        self.set_current_subtask_description(self._form_subtask_description(self.venv, *ds, task_type=task_type))

    def generate_task_description(self) -> str:
        subtasks = []
        task_type = self.venv.get_task_type()

        for objects in self.venv.get_subtask_objects():
            o1, o2 = objects if len(objects) == 2 else (objects[0], None)

            if task_type in TaskType.get_pattern_reach_task_types():
                subtasks.append(NaturalLanguage._form_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name_with_properties()))
            elif task_type in TaskType.get_pattern_press_task_types():
                subtasks.append(NaturalLanguage._form_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name()))
            else:
                subtasks.append(NaturalLanguage._form_subtask_description(self.venv, self.venv.get_task_type(), o1.get_name_with_properties(), "to " + o2.get_name_with_properties()))

        return ", ".join(subtasks)

    def generate_new_tasks(self, max_tasks=10, max_subtasks=3) -> List[str]:
        if self.venv.get_task_type() in TaskType.get_pattern_press_task_types():
            raise NotImplementedError()

        tuples = [("", self.venv)]
        for i in range(max_subtasks):
            tuples = self._generate_new_subtasks(tuples)
            if len(tuples) > max_tasks:
                tuples = self.rng.choice(tuples, max_tasks, replace=False)

        return [t[0] for t in tuples]
