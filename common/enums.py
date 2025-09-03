from enum import Enum


class ImageTriggerType(Enum):
    BLENDED_PATCH = "blended_patch"
    CHECKERBOARD = "checkerboard"


class ImageTriggerLocation(Enum):
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    CENTER = "center"


# --- Enums for Text Backdoors ---
class TextTriggerLocation(Enum):
    START = "start"
    END = "end"
    MIDDLE = "middle"
