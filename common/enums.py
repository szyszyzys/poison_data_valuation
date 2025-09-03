from enum import Enum


class ImageTriggerType(Enum):
    BLENDED_PATCH = "blended_patch"
    CHECKERBOARD = "checkerboard"
    NOISE = "noise"


class ImageTriggerLocation(Enum):
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    CENTER = "center"


# --- Enums for Text Backdoors ---
class TextTriggerLocation(Enum):
    START = "start"
    END = "end"
    MIDDLE = "middle"


class PoisonType(str, Enum):
    NONE = "none"
    BACKDOOR = "backdoor"
    LABEL_FLIP = "label_flip"


class LabelFlipMode(str, Enum):
    FIXED_TARGET = "fixed_target"
    RANDOM_DIFFERENT = "random_different"
