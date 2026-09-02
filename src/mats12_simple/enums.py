from enum import Enum


class DataSize(str, Enum):
    SMOKE = "smoke"
    TINY = "tiny"
    SMALL = "sm"
    MEDIUM = "md"
    FULL = "full"


class Modes(str, Enum):
    AUDIO_ONLY = "audio_only"
    AGREE = "aligned"
    CONFLICT = "audio_text_conflict"
    TEXT_ONLY_WRONG = "text_only_wrong"
    TEXT_ONLY_TRUTH = "text_only_truth"


class ModelVariant(str, Enum):
    PRETRAINED = "pretrained"
    INSTRUCT = "instruct"